import argparse
import json
import csv
import openai
import pandas as pd
import yaml
import sys
import os
import re
from tqdm import tqdm
import base64

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


def process_each_row_text(args, df, llm, mode, verbose=False, result_path=None):
    total_questions = 0
    correct_overall = 0
    accuracy_by_type = {}
    accuracy_by_subtype = {}
    accuracy_by_radius = {}

    if args['inference']['end_idx'] == 0:
        args['inference']['end_idx'] = len(df) - 1

    for index, row in tqdm(df.iloc[args['inference']['start_idx']:args['inference']['end_idx']].iterrows(),
                           total=args['inference']['end_idx'] - args['inference']['start_idx']):
        total_questions += 1
        prompt = row["Question"]

        # Append options.
        prompt += "\n\nOptions:\n" + row["All Options"]

        # Append additional data (with titles)
        try:
            titles = row["Data Titles"]  # Titles shou ld be a JSON array
            if isinstance(titles, str):
                titles = [titles]
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

        # Build the per-question result entry.
        question_result = {
            "entry_type": "question",
            "index": index,
            "modality": mode,
            "question": row["Question"],
            "selected_option": selected_option,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "type": row["Type"],
            "subtype": row["Subtype"],
            "radius": row["Radius"],
            "full_response": response,
        }

        # Save immediately to the result file if provided.
        with open(result_path, "a", encoding="utf-8") as file:  # 'a' mode for append
            file.write(json.dumps(question_result, ensure_ascii=False) + "\n")  # Append as JSONL

    aggregate_results = {
        "overall": {"correct": correct_overall, "total": total_questions},
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_subtype": accuracy_by_subtype,
        "accuracy_by_radius": accuracy_by_radius
    }
    return aggregate_results


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def process_each_row_image(args, df, llm, verbose=False, result_path=None):
    total_image_queries = 0
    overall_correct = 0
    image_types = ["heatmap", "heatmap_with_text", "heatmap_overlayed_on_map"]
    accuracy_by_image_type = {img_type: {"total": 0, "correct": 0} for img_type in image_types}

    if args['inference']['end_idx'] == 0:
        args['inference']['end_idx'] = len(df) - 1

    for index, row in tqdm(df.iloc[args['inference']['start_idx']:args['inference']['end_idx']].iterrows(),
                           total=args['inference']['end_idx'] - args['inference']['start_idx']):
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

            base64_image = encode_image(image_path)

            prompt = row["Question"]
            prompt += "\n\nOptions:\n" + row["All Options"]
            prompt += "Instruction: Analyze this image and answer the question. Think step by step before making a decision. Then, explicitly state your final choice after the special word 'Final Answer:'."

            payload = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
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

            # Build the per-image query result entry.
            image_result = {
                "entry_type": "image_query",
                "row_index": index,
                "image_type": img_type,
                "image_path": image_path,
                "selected_option": selected_option,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "full_response": response,
            }

            # Save immediately to the result file if provided.
            with open(result_path, "a", encoding="utf-8") as file:  # 'a' mode for append
                file.write(json.dumps(image_result, ensure_ascii=False) + "\n")  # Append as JSONL

    aggregate_results = {
        "overall": {"correct": overall_correct, "total": total_image_queries},
        "accuracy_by_image_type": accuracy_by_image_type
    }
    return aggregate_results


def main():
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print("Error reading config.yaml:", e)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run Q/A from CSV and query LLM.")
    parser.add_argument("--model", default="gpt-4o", help="Model name to use.")
    parser.add_argument("--use_url", action="store_true", help="Use internal URL to access the model.")
    parser.add_argument("--modality", choices=["text", "code", "image", "all"], default="text",
                        help="Specify whether to instruct the model to use text, code, image, or all modes.")
    parser.add_argument("--question_path", default="output/qa_data.csv", help="Path to the CSV file.")
    parser.add_argument("--result_path", default="result/eval_results_gpt-4o_text.jsonl", help="Path to the result file.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting question index.")
    parser.add_argument("--end_idx", type=int, default=0, help="Ending question index.")
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    config['models']['llm'] = args.model
    config['models']['use_url'] = args.use_url
    config['inference']['modality'] = args.modality
    config['inference']['question_path'] = args.question_path
    config['inference']['verbose'] = args.verbose
    config['inference']['start_idx'] = args.start_idx
    config['inference']['end_idx'] = args.end_idx

    llm = QueryLLM(config)
    df = pd.read_csv(args.question_path)
    modality = ["text", "code", "image"] if args.modality == "all" else [args.modality]

    if args.clean and os.path.exists(args.result_path):
        os.remove(args.result_path)  # Remove the file

    for mode in modality:
        print(f"\n********** Running mode: {mode} **********")

        if mode in ["text", "code"]:
            results = process_each_row_text(args, df, llm, mode, args.verbose, result_path=args.result_path)
            print(f"\nResults for mode '{mode}':")
        elif mode == "image":
            results = process_each_row_image(args, df, llm, args.verbose, result_path=args.result_path)
        else:
            continue

        # Save aggregated results immediately as individual JSON objects
        with open(args.result_path, "a") as outfile:
            aggregate_entry = {
                "entry_type": "aggregate",
                "modality": mode,
                "result": results
            }
            json.dump(aggregate_entry, outfile)
            outfile.write("\n")

        print(f"\nAggregated results for mode '{mode}' appended to {args.result_path}")


if __name__ == "__main__":
    main()
