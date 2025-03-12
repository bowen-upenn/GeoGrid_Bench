import argparse
import json
import csv
import openai
import pandas as pd
import yaml
import sys
import re
from tqdm import tqdm

from query_llm import QueryLLM
import utils


def parse_answer(response):
    """
    Extract the final answer from the LLM response after 'Final Answer:'.
    Consider it correct only if exactly one valid choice (a, b, c, or d) is found.
    """

    # Extract the portion after "Final Answer:"
    match = re.search(r"Final Answer:\s*(.*)", response, re.IGNORECASE)

    if not match:
        return None  # No "Final Answer:" found

    answer_text = match.group(1).strip()  # Extract answer part and strip whitespace

    # Find all occurrences of (a), (b), (c), or (d)
    options = re.findall(r"\(\s*[a-dA-D]\s*\)", answer_text)

    if len(options) == 1:  # Ensure exactly one valid answer
        return options[0].lower()  # Normalize case (e.g., "(A)" -> "(a)")

    return None


def update_accuracy(is_correct, counters, key):
    if key not in counters:
        counters[key] = {"total": 0, "correct": 0}
    counters[key]["total"] += 1
    if is_correct:
        counters[key]["correct"] += 1


def process_each_row_text(df, llm, mode, verbose=False):
    total_questions = 0
    correct_overall = 0
    accuracy_by_type = {}
    accuracy_by_subtype = {}
    accuracy_by_radius = {}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        total_questions += 1
        prompt = row["Question"]

        # Append options.
        prompt += "\n\nOptions:\n" + row["All Options"]

        # Append additional data (with titles)
        try:
            titles = row["Titles"]  # Titles should be a JSON array
        except Exception:
            titles = ["Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"]

        for i, col in enumerate(["Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"]):
            cell = row.get(col, None)
            if pd.notnull(cell) and str(cell).strip() != "":
                title = titles[i] if i < len(titles) else col
                prompt += f"\n\n{title}: {cell}"

        # Append instruction based on mode
        if mode == "text":
            prompt += "\n\nInstruction: Think step by step before making a decision. Then, explicitly state your final choice after the special word 'Final Answer:'. Please do NOT use programming code."
        if mode == "code":
            prompt += "\n\nInstruction: Think step by step before making a decision. Then, explicitly state your final choice after the special word 'Final Answer:'. Please use programming code to help you."

        if verbose:
            print(f"\n=== Question {total_questions} [{mode}] ===")
            print(prompt)

        # Query LLM
        response = llm.query_llm(step='inference', content=prompt, assistant=False, verbose=verbose)

        # Extract and validate answer
        selected_option = parse_answer(response)
        correct_answer = row["Correct Answer"]
        is_correct = (selected_option == correct_answer) if selected_option else False
        if is_correct:
            correct_overall += 1

        if verbose:
            if selected_option == correct_answer:
                print(f'{utils.Colors.OKGREEN}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
            else:
                print(f'{utils.Colors.FAIL}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')

        # Update accuracy metrics
        update_accuracy(is_correct, accuracy_by_type, row["Type"])
        update_accuracy(is_correct, accuracy_by_subtype, row["Subtype"])
        update_accuracy(is_correct, accuracy_by_radius, row["Radius"])

    return {
        "overall": {"correct": correct_overall, "total": total_questions},
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_subtype": accuracy_by_subtype,
        "accuracy_by_radius": accuracy_by_radius
    }


def process_each_row_image(df, llm, verbose=False):
    total_image_queries = 0
    overall_correct = 0
    image_types = ["heatmap", "heatmap_with_text", "heatmap_overlayed_on_map"]
    accuracy_by_image_type = {img_type: {"total": 0, "correct": 0} for img_type in image_types}

    for index, row in tqdm(df.iterrows(), total=len(df)):
        correct_answer = row["Correct Answer"]
        try:
            image_paths = json.loads(row["Image Paths"])
            if len(image_paths) != 3:
                continue
        except Exception:
            continue

        for i, image_path in enumerate(image_paths):
            total_image_queries += 1
            img_type = image_types[i]

            payload = [
                {"type": "text", "text": "Analyze this image and answer the question. Think step by step before making a decision. Then, explicitly state your final choice after the special word 'Final Answer:'."},
                {"type": "image_url", "image_url": {"url": image_path}}
            ]

            if verbose:
                print(f"\n=== Image Query for row {index} [{img_type}] ===")
                print(f"Image path: {image_path}")

            response = llm.query_llm(step='inference', content=payload, assistant=False, verbose=verbose)

            selected_option = parse_answer(response)
            is_correct = (selected_option == correct_answer) if selected_option else False
            if is_correct:
                overall_correct += 1

            if verbose:
                if selected_option == correct_answer:
                    print(f'{utils.Colors.OKGREEN}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
                else:
                    print(f'{utils.Colors.FAIL}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')

            update_accuracy(is_correct, accuracy_by_image_type, img_type)

    return {
        "overall": {"correct": overall_correct, "total": total_image_queries},
        "accuracy_by_image_type": accuracy_by_image_type
    }


def main():
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print("Error reading config.yaml:", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run Q/A from CSV and query LLM.")
    parser.add_argument("--model", default="gpt-4", help="Model name to use.")
    parser.add_argument("--mode", choices=["text", "code", "image", "all"], default="text",
                        help="Specify whether to instruct the model to use text, code, image, or all modes.")
    parser.add_argument("--csv", default="output/qa_data.csv", help="Path to the CSV file (tab-delimited).")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    config['models']['llm'] = args.model
    config['inference']['mode'] = args.mode
    config['inference']['csv'] = args.csv
    config['inference']['verbose'] = args.verbose

    llm = QueryLLM(config)
    df = pd.read_csv(args.csv)

    final_results = {}

    modes = ["text", "code", "image"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n********** Running mode: {mode} **********")
        if mode in ["text", "code"]:
            results = process_each_row_text(df, llm, mode, args.verbose)
            final_results[mode] = results
        elif mode == "image":
            results = process_each_row_image(df, llm, args.verbose)
            final_results[mode] = results

    output_filename = "output/eval_results.json"
    try:
        with open(output_filename, "w") as outfile:
            json.dump(final_results, outfile, indent=4)
        print(f"\nFinal results saved to {output_filename}")
    except Exception as e:
        print("Error saving final results:", e)


if __name__ == "__main__":
    main()
