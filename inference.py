import argparse
import json
import openai
import pandas as pd
import yaml
import sys
import os
import re
import base64
from tqdm import tqdm
from PIL import Image

from query_llm import QueryLLM
import utils
import prompts


def parse_answer(response):
    # unchanged
    segments = re.split(
        r"(?i)(?:####Final Answer:|####|final answer(?: is|:)?|I choose|I select|option|choice)",
        response
    )
    tail = segments[-1].strip()
    pattern = (
        r"(?:"
          r"\(\s*([A-Da-d])\s*\)"
        r"|\[\s*([A-Da-d])\s*\]"
        r"|\\box(?:ed)?\{\s*([A-Da-d])\s*\}"
        r"|\b([A-Da-d])\b"
        r")"
    )
    m = re.search(pattern, tail)
    if not m:
        return None
    letter = next(g for g in m.groups() if g)
    return f"({letter.lower()})"


def update_accuracy(is_correct, counters, key):
    if key not in counters:
        counters[key] = {"total": 0, "correct": 0}
    counters[key]["total"] += 1
    if is_correct:
        counters[key]["correct"] += 1


def adjust_start_end_index(start, end, result_path):
    try:
        with open(result_path, 'r') as f:
            lines = f.read().splitlines()
            for line in reversed(lines):
                record = json.loads(line)
                if "index" in record:
                    if start <= record["index"] < end:
                        start = record["index"] + 1
                        print(f"Resuming from index {start}")
                        break
                elif "row_index" in record:
                    if start <= record["row_index"] < end:
                        start = record["row_index"] + 1
                        print(f"Resuming from index {start}")
                        break
            if start == end:
                print("All questions have been processed. Exiting.")
                sys.exit(0)
    except Exception as e:
        print(f"Failed to load start_idx from result file: {e}")
        start = 0
    return start, end

def process_each_row_text(args, df, llm, mode, verbose=False, result_path=None, few_shots=False):

    assert result_path is not None, "result_path must be provided for saving results"
    # only tabular rows
    tab_df = df[df["data_modality"] == "tabular"].reset_index(drop=True)

    start = int(args["inference"]["start_idx"])
    end = int(args["inference"]["end_idx"])

    if end == -1 or end > len(tab_df):
        end = len(tab_df)

    if args["resume"] and os.path.exists(result_path):
        start, end = adjust_start_end_index(start, end, result_path)

    total_questions = 0
    correct_overall = 0
    accuracy_by_type = {}
    accuracy_by_radius = {}

    for idx, row in tqdm(tab_df.iloc[start:end].iterrows(), total=end-start):
        total_questions += 1

        prompt = ""
        if few_shots:
            template = row["Template Question"]
            prompt += prompts.few_shot_examples(template)

        prompt += row["Question"]
        prompt += "\n\nOptions:\n" + row["All Options"]

        if row["Type"] == "trend":
            prompt += "\n\nWe define the magnitude of change as follows:"
            prompt += "\n- Slightly: <10%."
            prompt += "\n- Moderately: 10–50%."
            prompt += "\n- Significantly: >50%."

        # Append data tables
        titles = row.get("Data Titles")
        if isinstance(titles, str):
            try:
                titles = json.loads(titles)
            except:
                titles = [titles]
        if not isinstance(titles, list):
            titles = []

        for i, col in enumerate([f"Data {i}" for i in range(1,9)], start=1):
            cell = row.get(col)
            if pd.notnull(cell) and str(cell).strip() != "":
                title = titles[i-1] if i-1 < len(titles) else col
                prompt += f"\n\n{title}: {cell}"

        # Instruction by mode
        if mode == "text":
            prompt += "\n\nInstruction: Think step by step before making a decision. Then, explicitly state your final choice after the special word '####Final Answer:' including (a), (b), (c), or (d). Please do NOT use programming code."
        if mode == "code":
            prompt += ("\n\nInstruction: Please write a Python programming code to answer the question. You must include print statements in your code to show the final answer using '####Final Answer:' including (a), (b), (c), or (d). "
                       "Do not just describe the answer. Instead, think step by step and show me the complete code. ")
            llm.create_a_thread()

        if verbose:
            print(f"\n=== [{mode}] Q{total_questions} ===\n{prompt}\n")

        response = llm.query_llm(step="inference", content=prompt, mode=mode, assistant=False, verbose=verbose)
        selected = parse_answer(response)
        correct = (selected == row["Correct Answer"])
        if correct:
            correct_overall += 1

        update_accuracy(correct, accuracy_by_type, row["Type"])
        update_accuracy(correct, accuracy_by_radius, row["Radius"])

        # result entry
        out = {
            "question_id": row["Question ID"],
            "entry_type": "question",
            "index": idx,
            "modality": mode,
            "question": row["Question"],
            "selected_option": selected,
            "correct_answer": row["Correct Answer"],
            "is_correct": correct,
            "type": row["Type"],
            "radius": row["Radius"],
            "full_response": response,
        }
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "overall": {"correct": correct_overall, "total": total_questions},
        "accuracy_by_type": accuracy_by_type,
        "accuracy_by_radius": accuracy_by_radius
    }


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def process_each_row_image(args, df, llm, verbose=False, result_path=None):

    assert result_path is not None, "result_path must be provided for saving results"
    # all non-tabular rows
    img_df = df[df["data_modality"] != "tabular"].reset_index(drop=True)
    
    start = int(args["inference"]["start_idx"])
    end = int(args["inference"]["end_idx"])

    if end == -1 or end > len(img_df):
        end = len(img_df)

    if args["resume"] and os.path.exists(result_path):
        start, end = adjust_start_end_index(start, end, result_path)

    total = 0
    correct_overall = 0
    accuracy_by_image_type = {}

    # init counters
    for mod in img_df["data_modality"].unique():
        accuracy_by_image_type[mod] = {"total": 0, "correct": 0}

    for idx, row in tqdm(img_df.iloc[start:end].iterrows(), total=end-start):
        total += 1
        img_type = row["data_modality"]
        image_path = row["image_path"]
        correct_answer = row["Correct Answer"]

        prompt = row["Question"] + "\n\nOptions:\n" + row["All Options"]
        if row["Type"] == "trend":
            prompt += "\n\nWe define the magnitude of change as follows:"
            prompt += "\n- Slightly: <10%."
            prompt += "\n- Moderately: 10–50%."
            prompt += "\n- Significantly: >50%."

        prompt += "\nInstruction: Analyze this image and answer. Think step by step. Then state 'Final Answer:' with (a), (b), (c), or (d)."

        base64_image = encode_image(image_path)
        model = args["models"]["llm"]

        # build messages based on model
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
        elif re.search(r'InternVL', model, flags=re.IGNORECASE):
            from internvl_util import load_image, build_transform
            messages = {
                'image': load_image(image_path, max_num=12),
                'text': prompt
            }
        else:
            # ---------------------------
            # Llama
            image = Image.open(image_path)
            messages = {'image': image, 'text': prompt}

        if verbose:
            print(f"\n=== [image:{img_type}] row {idx} === {image_path}\n")

        response = llm.query_llm(step="inference", content=messages, mode="image", assistant=False, verbose=verbose)
        selected = parse_answer(response)
        is_correct = (selected == correct_answer)
        if is_correct:
            correct_overall += 1

        update_accuracy(is_correct, accuracy_by_image_type, img_type)

        out = {
            "question_id": row["Question ID"],
            "entry_type": "image_query",
            "index": idx,
            "modality": img_type,
            "image_path": image_path,
            "selected_option": selected,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "full_response": response,
        }
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    return {
        "overall": {"correct": correct_overall, "total": total},
        "accuracy_by_image_type": accuracy_by_image_type
    }


def main():
    # load config
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print("Error reading config.yaml:", e)
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--use_url", action="store_true")
    p.add_argument("--modality", choices=["text","code","image","all"], default="text")
    p.add_argument("--question_path", default="data/benchmark/qa_data.csv")
    p.add_argument("--result_path", default="result/eval_results.jsonl")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--clean", dest='clean', action='store_true', help='Remove existing data files and start clean')
    p.add_argument("--few_shots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # override config
    config["models"]["llm"] = args.model
    config["models"]["use_url"] = args.use_url
    config["inference"]["modality"] = args.modality
    config["inference"]["question_path"] = args.question_path
    config["inference"]["start_idx"] = args.start_idx
    config["inference"]["end_idx"] = args.end_idx
    config["inference"]["verbose"] = args.verbose
    config["inference"]["few_shots"] = args.few_shots
    config['resume'] = args.resume

    if args.clean and os.path.exists(args.result_path):
        os.remove(args.result_path)  # Remove the file

    llm = QueryLLM(config, step="inference")
    df = pd.read_csv(args.question_path)

    modes = ["text","code","image"] if args.modality=="all" else [args.modality]
    for mode in modes:
        print(f"\n===== Running {mode} mode =====")
        if mode in ("text","code"):
            agg = process_each_row_text(config, df, llm, mode, args.verbose, args.result_path, args.few_shots)
            if args.few_shots:
                args.result_path = f"{args.result_path[:-6]}_fs.jsonl"
        else:
            agg = process_each_row_image(config, df, llm, args.verbose, args.result_path)

        # write aggregate
        with open(args.result_path, "a") as f:
            entry = {"entry_type":"aggregate","modality":mode,"result":agg}
            f.write(json.dumps(entry) + "\n")

        print(f"-> Finished {mode}, results appended to {args.result_path}")


if __name__ == "__main__":
    main()



# import argparse
# import json
# import csv
# import openai
# import pandas as pd
# import yaml
# import sys
# import os
# import re
# from tqdm import tqdm
# import base64
# import os
# from PIL import Image
#
# # from google import genai  # Gemini has conflicting requirements of the environment with OpenAI
# # from google.genai import types
# import PIL.Image
#
# from query_llm import QueryLLM
# import utils
#
#
# def parse_answer(response):
#     """
#     Extracts a single choice (a, b, c, or d) from the response. First it
#     splits on any of several "lead-in" markers, then scans the
#     trailing segment for one valid letter choice in various formats.
#     If no valid choice is found, returns None.
#     """
#     # split on lead-in markers (case-insensitive)
#     segments = re.split(
#         r"(?i)(?:####Final Answer:|####|final answer(?: is|:)?|I choose|I select|option|choice)",
#         response
#     )
#     tail = segments[-1].strip()
#
#     # look for (a), [a], \box{a}, \boxed{a}, or standalone a–d as a whole word
#     pattern = (
#         r"(?:"
#           r"\(\s*([A-Da-d])\s*\)"       # (a)
#         r"|\[\s*([A-Da-d])\s*\]"       # [a]
#         r"|\\box(?:ed)?\{\s*([A-Da-d])\s*\}"  # \box{a} or \boxed{a}
#         r"|\b([A-Da-d])\b"             # standalone a
#         r")"
#     )
#     m = re.search(pattern, tail)
#     if not m:
#         return None
#
#     # whichever group matched, pick that letter
#     letter = next(group for group in m.groups() if group)
#     return f"({letter.lower()})"
#
#
# def update_accuracy(is_correct, counters, key):
#     if key not in counters:
#         counters[key] = {"total": 0, "correct": 0}
#     counters[key]["total"] += 1
#     if is_correct:
#         counters[key]["correct"] += 1
#
#
# def process_each_row_text(args, df, llm, mode, verbose=False, result_path=None):
#     total_questions = 0
#     correct_overall = 0
#     accuracy_by_type = {}
#     accuracy_by_subtype = {}
#     accuracy_by_radius = {}
#
#     if int(args['inference']['end_idx']) == -1:
#         args['inference']['end_idx'] = len(df) - 1
#
#     for index, row in tqdm(df.iloc[int(args['inference']['start_idx']):int(args['inference']['end_idx'])].iterrows(),
#                            total=int(args['inference']['end_idx']) - int(args['inference']['start_idx'])):
#         total_questions += 1
#         prompt = row["Question"]
#
#         # Append options.
#         prompt += "\n\nOptions:\n" + row["All Options"]
#         if row['Type'] == "trend":
#             # Explain what slightly, moderately, and significantly mean.
#             prompt += "\n\nWe define the magnitude of change as follows:"
#             prompt += "\n\n- Slightly: the percentage change is less than 10%."
#             prompt += "\n- Moderately: the percentage change is between 10% and 50%."
#             prompt += "\n- Significantly: the percentage change is greater than 50%."
#
#         # Append additional data (with titles)
#         try:
#             titles = row["Data Titles"]  # Titles shou ld be a JSON array
#             if isinstance(titles, str):
#                 titles = [titles]
#         except Exception:
#             titles = ["Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"]
#
#         for i, col in enumerate(["Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"]):
#             cell = row.get(col, None)
#             if pd.notnull(cell) and str(cell).strip() != "":
#                 title = titles[i] if i < len(titles) else col
#                 prompt += f"\n\n{title}: {cell}"
#
#         # Append instruction based on mode
#         if mode == "text":
#             prompt += "\n\nInstruction: Think step by step before making a decision. Then, explicitly state your final choice after the special word '####Final Answer:' including (a), (b), (c), or (d). Please do NOT use programming code."
#         if mode == "code":
#             prompt += ("\n\nInstruction: Please write a Python programming code to answer the question. You must include print statements in your code to show the final answer using '####Final Answer:' including (a), (b), (c), or (d). "
#                        "Do not just describe the answer. Instead, think step by step and show me the complete code. ")
#             llm.create_a_thread()
#
#         if verbose:
#             print(f"\n=== Question {total_questions} [{mode}] ===")
#             print(prompt)
#
#         # Query LLM
#         response = llm.query_llm(step='inference', content=prompt, mode=mode, assistant=False, verbose=verbose)
#
#         # Extract and validate answer
#         selected_option = parse_answer(response)
#         correct_answer = row["Correct Answer"]
#         is_correct = (selected_option == correct_answer) if selected_option else False
#         if is_correct:
#             correct_overall += 1
#
#         if verbose:
#             if selected_option == correct_answer:
#                 print(f'{utils.Colors.OKGREEN}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
#             else:
#                 print(f'{utils.Colors.FAIL}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
#
#         # Update accuracy metrics
#         update_accuracy(is_correct, accuracy_by_type, row["Type"])
#         update_accuracy(is_correct, accuracy_by_subtype, row["Subtype"])
#         update_accuracy(is_correct, accuracy_by_radius, row["Radius"])
#
#         # Build the per-question result entry.
#         question_result = {
#             "question_id": row["Question ID"],
#             "entry_type": "question",
#             "index": index,
#             "modality": mode,
#             "question": row["Question"],
#             "selected_option": selected_option,
#             "correct_answer": correct_answer,
#             "is_correct": is_correct,
#             "type": row["Type"],
#             "subtype": row["Subtype"],
#             "radius": row["Radius"],
#             "full_response": response,
#         }
#
#         # Save immediately to the result file if provided.
#         os.makedirs(os.path.dirname(result_path), exist_ok=True)
#
#         with open(result_path, "a", encoding="utf-8") as file:  # 'a' mode for append
#             file.write(json.dumps(question_result, ensure_ascii=False) + "\n")  # Append as JSONL
#
#     aggregate_results = {
#         "overall": {"correct": correct_overall, "total": total_questions},
#         "accuracy_by_type": accuracy_by_type,
#         "accuracy_by_subtype": accuracy_by_subtype,
#         "accuracy_by_radius": accuracy_by_radius
#     }
#     return aggregate_results
#
#
# # Function to encode the image
# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")
#
#
# def process_each_row_image(args, df, llm, verbose=False, result_path=None):
#     total_image_queries = 0
#     overall_correct = 0
#     image_types = ["heatmap", "heatmap_with_text", "heatmap_overlayed_on_map"]
#     accuracy_by_image_type = {img_type: {"total": 0, "correct": 0} for img_type in image_types}
#     model = args['models']['llm']
#
#     if args['inference']['end_idx'] == -1:
#         args['inference']['end_idx'] = len(df) - 1
#
#     for index, row in tqdm(df.iloc[args['inference']['start_idx']:args['inference']['end_idx']].iterrows(),
#                            total=args['inference']['end_idx'] - args['inference']['start_idx']):
#         correct_answer = row["Correct Answer"]
#         try:
#             image_paths = json.loads(row["Image Paths"])
#             if len(image_paths) != 3:
#                 continue
#         except Exception:
#             continue
#
#         for i, image_path in enumerate(image_paths):
#             total_image_queries += 1
#             img_type = image_types[i]
#
#             base64_image = encode_image(image_path)
#
#             prompt = row["Question"]
#             prompt += "\n\nOptions:\n" + row["All Options"]
#             if row['Type'] == "trend":
#                 # Explain what slightly, moderately, and significantly mean.
#                 prompt += "\n\nWe define the magnitude of change as follows:"
#                 prompt += "\n\n- Slightly: the percentage change is less than 10%."
#                 prompt += "\n- Moderately: the percentage change is between 10% and 50%."
#                 prompt += "\n- Significantly: the percentage change is greater than 50%."
#             prompt += "Instruction: Analyze this image and answer the question. Think step by step before making a decision. Then, explicitly state your final choice after the special word 'Final Answer:'."
#
#             # Branch based on the model type for handling images
#             if utils.check_openai_models(args) or re.search(r'llama-4', model, flags=re.IGNORECASE):
#                 # ---------------------------
#                 # OpenAI API
#                 messages = [
#                     {"type": "text", "text": prompt},
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
#                 ]
#             elif re.search(r'gemini', model, flags=re.IGNORECASE):
#                 # ---------------------------
#                 # Google Gemini API
#                 messages = [
#                     prompt,
#                     types.Part.from_bytes(data=base64_image, mime_type="image/png")
#                 ]
#             elif re.search(r'claude', model, flags=re.IGNORECASE):
#                 # ---------------------------
#                 # Anthropic Claude API
#                 messages = [
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "image",
#                                 "source": {
#                                     "type": "base64",
#                                     "media_type": "image/png",
#                                     "data": base64_image  # assuming this is your base64-encoded string
#                                 }
#                             },
#                             {
#                                 "type": "text",
#                                 "text": prompt  # assuming this is your prompt text
#                             }
#                         ]
#                     }
#                 ]
#             elif re.search(r'Qwen', model, flags=re.IGNORECASE):
#                 messages = [
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "image",
#                                 "image": f"data:image;base64,{base64_image}",
#                             },
#                             {"type": "text", "text": prompt},
#                         ],
#                     }
#                 ]
#             elif re.search(r'Phi', model, flags=re.IGNORECASE):
#                 image = Image.open(image_path)
#                 messages = {'image': image, 'text': prompt}
#             else:
#                 # ---------------------------
#                 # Llama
#                 image = Image.open(image_path)
#                 messages = {'image': image, 'text': prompt}
#
#             if verbose:
#                 print(f"\n=== Image Query for row {index} [{img_type}] ===")
#                 print(f"Image path: {image_path}")
#
#             response = llm.query_llm(step='inference', content=messages, mode='image', assistant=False, verbose=verbose)
#
#             selected_option = parse_answer(response)
#             is_correct = (selected_option == correct_answer) if selected_option else False
#             if is_correct:
#                 overall_correct += 1
#
#             if verbose:
#                 if selected_option == correct_answer:
#                     print(f'{utils.Colors.OKGREEN}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
#                 else:
#                     print(f'{utils.Colors.FAIL}{"Model Answer:"}{selected_option}{" Correct Answer:"}{row["Correct Answer"]}{utils.Colors.ENDC}')
#
#             update_accuracy(is_correct, accuracy_by_image_type, img_type)
#
#             # Build the per-image query result entry.
#             image_result = {
#                 "question_id": row["Question ID"],
#                 "entry_type": "image_query",
#                 "index": index,
#                 "image_type": img_type,
#                 "image_path": image_path,
#                 "selected_option": selected_option,
#                 "correct_answer": correct_answer,
#                 "is_correct": is_correct,
#                 "full_response": response,
#             }
#
#             os.makedirs(os.path.dirname(result_path), exist_ok=True)
#             # Save immediately to the result file if provided.
#             with open(result_path, "a", encoding="utf-8") as file:  # 'a' mode for append
#                 file.write(json.dumps(image_result, ensure_ascii=False) + "\n")  # Append as JSONL
#
#     aggregate_results = {
#         "overall": {"correct": overall_correct, "total": total_image_queries},
#         "accuracy_by_image_type": accuracy_by_image_type
#     }
#     return aggregate_results
#
#
# def main():
#     try:
#         with open('config.yaml', 'r') as file:
#             config = yaml.safe_load(file)
#     except Exception as e:
#         print("Error reading config.yaml:", e)
#         sys.exit(1)
#
#     parser = argparse.ArgumentParser(description="Run Q/A from CSV and query LLM.")
#     parser.add_argument("--model", default="gpt-4o", help="Model name to use.")
#     parser.add_argument("--use_url", action="store_true", help="Use internal URL to access the model.")
#     parser.add_argument("--modality", choices=["text", "code", "image", "all"], default="text",
#                         help="Specify whether to instruct the model to use text, code, image, or all modes.")
#     parser.add_argument("--question_path", default="output/benchmark/qa_data.csv", help="Path to the CSV file.")
#     parser.add_argument("--result_path", default="result/eval_results_gpt-4o_text.jsonl", help="Path to the result file.")
#     parser.add_argument("--start_idx", type=int, default=-1, help="Starting question index; -1 to resume from last run.")
#     parser.add_argument("--end_idx", type=int, default=-1, help="Ending question index. -1 means iterate till the end.")
#     parser.add_argument("--resume", action="store_true", help="Resume from the last index.")
#     parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing data files and start clean')
#     parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
#     args = parser.parse_args()
#
#     # If start_idx is -1, try to get it from the last "index" entry in result_path
#     if (args.start_idx == -1 or args.resume) and os.path.exists(args.result_path):
#         try:
#             with open(args.result_path, 'r') as f:
#                 lines = f.read().splitlines()
#                 for line in reversed(lines):
#                     record = json.loads(line)
#                     if "index" in record:
#                         if args.start_idx <= record["index"] < args.end_idx:
#                             args.start_idx = record["index"] + 1
#                             print(f"Resuming from index {args.start_idx}")
#                             break
#                     elif "row_index" in record:
#                         if args.start_idx <= record["row_index"] < args.end_idx:
#                             args.start_idx = record["row_index"] + 1
#                             print(f"Resuming from index {args.start_idx}")
#                             break
#                 if args.start_idx == args.end_idx:
#                     print("All questions have been processed. Exiting.")
#                     sys.exit(0)
#         except Exception as e:
#             print(f"Failed to load start_idx from result file: {e}")
#             args.start_idx = 0
#
#     config['models']['llm'] = args.model
#     config['models']['use_url'] = args.use_url
#     config['inference']['modality'] = args.modality
#     config['inference']['question_path'] = args.question_path
#     config['inference']['verbose'] = args.verbose
#     config['inference']['start_idx'] = int(args.start_idx)
#     config['inference']['end_idx'] = int(args.end_idx)
#
#     llm = QueryLLM(config, step='inference')
#     df = pd.read_csv(args.question_path)
#     modality = ["text", "code", "image"] if args.modality == "all" else [args.modality]
#
#     # if args.clean and os.path.exists(args.result_path):
#     #     os.remove(args.result_path)  # Remove the file
#
#     for mode in modality:
#         print(f"\n********** Running mode: {mode} **********")
#
#         if mode in ["text", "code"]:
#             results = process_each_row_text(config, df, llm, mode, args.verbose, result_path=args.result_path)
#             print(f"\nResults for mode '{mode}':")
#         elif mode == "image":
#             results = process_each_row_image(config, df, llm, args.verbose, result_path=args.result_path)
#         else:
#             continue
#
#         # Save aggregated results immediately as individual JSON objects
#         with open(args.result_path, "a") as outfile:
#             aggregate_entry = {
#                 "entry_type": "aggregate",
#                 "modality": mode,
#                 "result": results
#             }
#             json.dump(aggregate_entry, outfile)
#             outfile.write("\n")
#
#         print(f"\nAggregated results for mode '{mode}' appended to {args.result_path}")
#
#
# if __name__ == "__main__":
#     main()
