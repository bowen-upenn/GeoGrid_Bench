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
import os
from PIL import Image

# from google import genai  # Gemini has conflicting requirements of the environment with OpenAI
# from google.genai import types
import PIL.Image

from query_llm import QueryLLM
import utils


def parse_answer(response):
    """
    Extracts a single choice (a, b, c, or d) from the response. First it
    splits on any of several "lead-in" markers, then scans the
    trailing segment for one valid letter choice in various formats.
    If no valid choice is found, returns None.
    """
    # split on lead-in markers (case-insensitive)
    segments = re.split(
        r"(?i)(?:####Final Answer:|####|final answer(?: is|:)?|I choose|I select|option|choice)",
        response
    )
    tail = segments[-1].strip()

    # look for (a), [a], \box{a}, \boxed{a}, or standalone a–d as a whole word
    pattern = (
        r"(?:"
          r"\(\s*([A-Da-d])\s*\)"       # (a)
        r"|\[\s*([A-Da-d])\s*\]"       # [a]
        r"|\\box(?:ed)?\{\s*([A-Da-d])\s*\}"  # \box{a} or \boxed{a}
        r"|\b([A-Da-d])\b"             # standalone a
        r")"
    )
    m = re.search(pattern, tail)
    if not m:
        return None

    # whichever group matched, pick that letter
    letter = next(group for group in m.groups() if group)
    return f"({letter.lower()})"


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

    if int(args['inference']['end_idx']) == -1:
        args['inference']['end_idx'] = len(df) - 1

    for index, row in tqdm(df.iloc[int(args['inference']['start_idx']):int(args['inference']['end_idx'])].iterrows(),
                           total=int(args['inference']['end_idx']) - int(args['inference']['start_idx'])):
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
            prompt += "\n\nInstruction: Think step by step before making a decision. Then, explicitly state your final choice after the special word '####Final Answer:' including (a), (b), (c), or (d). Please do NOT use programming code."
        if mode == "code":
            prompt += "\n\nInstruction: Please write a Python programming code to answer the question. Show me the complete code. You must include print statements at the end of your code to show the final answer using '####Final Answer:' including (a), (b), (c), or (d)."
            llm.create_a_thread()

        if verbose:
            print(f"\n=== Question {total_questions} [{mode}] ===")
            print(prompt)

        # Query LLM
        response = llm.query_llm(step='inference', content=prompt, mode=mode, assistant=(mode=='code'), verbose=verbose)

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
            "question_id": row["Question ID"],
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
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

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
    model = args['models']['llm']

    if args['inference']['end_idx'] == -1:
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

            # Branch based on the model type for handling images
            if utils.check_openai_models(args) or re.search(r'llama-4', model, flags=re.IGNORECASE):
                # ---------------------------
                # OpenAI API
                messages = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            elif re.search(r'gemini', model, flags=re.IGNORECASE):
                # ---------------------------
                # Google Gemini API
                messages = [
                    prompt, 
                    types.Part.from_bytes(data=base64_image, mime_type="image/png")
                ]
            elif re.search(r'claude', model, flags=re.IGNORECASE):
                # ---------------------------
                # Anthropic Claude API
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image  # assuming this is your base64-encoded string
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt  # assuming this is your prompt text
                            }
                        ]
                    }
                ]
            elif re.search(r'Qwen', model, flags=re.IGNORECASE):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": f"data:image;base64,{base64_image}",
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            elif re.search(r'Phi', model, flags=re.IGNORECASE):
                image = Image.open(image_path)
                messages = {'image': image, 'text': prompt}
            else:
                # ---------------------------
                # Llama
                image = Image.open(image_path)
                messages = {'image': image, 'text': prompt}

            if verbose:
                print(f"\n=== Image Query for row {index} [{img_type}] ===")
                print(f"Image path: {image_path}")

            response = llm.query_llm(step='inference', content=messages, mode='image', assistant=False, verbose=verbose)

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
                "question_id": row["Question ID"],
                "entry_type": "image_query",
                "index": index,
                "image_type": img_type,
                "image_path": image_path,
                "selected_option": selected_option,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "full_response": response,
            }

            os.makedirs(os.path.dirname(result_path), exist_ok=True)
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
    parser.add_argument("--question_path", default="output/benchmark/qa_data.csv", help="Path to the CSV file.")
    parser.add_argument("--result_path", default="result/eval_results_gpt-4o_text.jsonl", help="Path to the result file.")
    parser.add_argument("--start_idx", type=int, default=-1, help="Starting question index; -1 to resume from last run.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Ending question index. -1 means iterate till the end.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last index.")
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    args = parser.parse_args()

    # If start_idx is -1, try to get it from the last "index" entry in result_path
    if (args.start_idx == -1 or args.resume) and os.path.exists(args.result_path):
        try:
            with open(args.result_path, 'r') as f:
                lines = f.read().splitlines()
                for line in reversed(lines):
                    record = json.loads(line)
                    if "index" in record:
                        if args.start_idx <= record["index"] < args.end_idx:
                            args.start_idx = record["index"] + 1
                            print(f"Resuming from index {args.start_idx}")
                            break
                    elif "row_index" in record:
                        if args.start_idx <= record["row_index"] < args.end_idx:
                            args.start_idx = record["row_index"] + 1
                            print(f"Resuming from index {args.start_idx}")
                            break
                if args.start_idx == args.end_idx:
                    print("All questions have been processed. Exiting.")
                    sys.exit(0)
        except Exception as e:
            print(f"Failed to load start_idx from result file: {e}")
            args.start_idx = 0

    config['models']['llm'] = args.model
    config['models']['use_url'] = args.use_url
    config['inference']['modality'] = args.modality
    config['inference']['question_path'] = args.question_path
    config['inference']['verbose'] = args.verbose
    config['inference']['start_idx'] = int(args.start_idx)
    config['inference']['end_idx'] = int(args.end_idx)

    llm = QueryLLM(config, step='inference')
    df = pd.read_csv(args.question_path)
    modality = ["text", "code", "image"] if args.modality == "all" else [args.modality]

    # if args.clean and os.path.exists(args.result_path):
    #     os.remove(args.result_path)  # Remove the file

    for mode in modality:
        print(f"\n********** Running mode: {mode} **********")

        if mode in ["text", "code"]:
            results = process_each_row_text(config, df, llm, mode, args.verbose, result_path=args.result_path)
            print(f"\nResults for mode '{mode}':")
        elif mode == "image":
            results = process_each_row_image(config, df, llm, args.verbose, result_path=args.result_path)
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
