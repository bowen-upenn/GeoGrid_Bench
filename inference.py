import argparse
import json
import csv
import openai
import pandas as pd


def query_llm(prompt, api_key, model="gpt-4o", max_tokens=1000, verbose=False):
    """
    Query the LLM with the given prompt.
    """
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    answer = response.choices[0].message.content
    if verbose:
        print(f"LLM Response: {answer}")
    return answer


def parse_answer(response):
    """
    Parse the LLM response to extract one of the options: (a), (b), (c), or (d).
    """
    for option in ['(a)', '(b)', '(c)', '(d)']:
        if option in response:
            return option
    return None


def main():
    parser = argparse.ArgumentParser(description="Run Q/A from CSV and query LLM.")
    parser.add_argument("--mode", choices=["text", "code", "image"], default="text",
                        help="Specify whether to instruct the model to use text, code, or image.")
    parser.add_argument("--csv", default="output/qa_data.csv", help="Path to the CSV file (tab-delimited).")
    args = parser.parse_args()

    # Initialize counters for accuracy calculations.
    total_questions = 0
    correct_overall = 0
    accuracy_by_type = {}
    accuracy_by_subtype = {}
    accuracy_by_radius = {}

    # Load OpenAI API key.
    with open("openai_key.txt", "r") as key_file:
        api_key = key_file.read().strip()

    df = pd.read_csv(args.csv)  # Read one line at a time

    # Process each row.
    for index, row in df.iterrows():
    # with open(args.csv, "r", newline="", encoding="utf-8") as file:
    #     reader = csv.reader(file)  # Use csv.DictReader(file) for dictionary format
    #     for row in reader:
        total_questions += 1
        print('row', row)

        # Build the prompt starting with the filled question template.
        prompt = row["Question"]
        print('prompt', prompt, 'END')

        # Append the available options (assumed to be a JSON list string).
        try:
            options = json.loads(row["All Options"])
            prompt += "\n\nOptions:\n" + "\n".join(options)
        except Exception as e:
            prompt += "\n\nOptions: " + str(row["All Options"])

        # Append any additional data fields that are non-empty.
        for col in ["Data 1", "Data 2", "Data 3", "Data 4", "Data 5", "Data 6", "Data 7", "Data 8"]:
            cell = row.get(col, None)
            if pd.notnull(cell) and str(cell).strip() != "":
                # Use the column header as the title.
                prompt += f"\n\n{col}: {cell}"

        # Append an instruction based on the selected mode.
        if args.mode == "text":
            prompt += ("\n\nInstruction: Provide your answer in plain text by selecting one of the options (a) to (d).")
        elif args.mode == "code":
            prompt += ("\n\nInstruction: Provide your answer as a code snippet. Include comments explaining your reasoning, "
                       "and indicate clearly which option (a) to (d) you select.")
        elif args.mode == "image":
            prompt += ("\n\nInstruction: Provide your answer using an image that illustrates your analysis. Clearly "
                       "indicate your selected option (a) to (d) in the image.")

        print(f"\n=== Question {total_questions} ===")
        print(prompt)

        # Query the LLM.
        response = query_llm(prompt, api_key)

        # Parse the answer to extract the chosen option.
        selected_option = parse_answer(response)
        print("Selected Option:", selected_option)

        # Check correctness.
        correct_answer = row["Correct Answer"]
        is_correct = (selected_option == correct_answer)
        if is_correct:
            correct_overall += 1

        # Update accuracy metrics for each category.
        # Type
        type_val = row["Type"]
        if type_val not in accuracy_by_type:
            accuracy_by_type[type_val] = {"total": 0, "correct": 0}
        accuracy_by_type[type_val]["total"] += 1
        if is_correct:
            accuracy_by_type[type_val]["correct"] += 1

        # Subtype
        subtype_val = row["Subtype"]
        if subtype_val not in accuracy_by_subtype:
            accuracy_by_subtype[subtype_val] = {"total": 0, "correct": 0}
        accuracy_by_subtype[subtype_val]["total"] += 1
        if is_correct:
            accuracy_by_subtype[subtype_val]["correct"] += 1

        # Radius
        radius_val = row["Radius"]
        if radius_val not in accuracy_by_radius:
            accuracy_by_radius[radius_val] = {"total": 0, "correct": 0}
        accuracy_by_radius[radius_val]["total"] += 1
        if is_correct:
            accuracy_by_radius[radius_val]["correct"] += 1

    # Calculate overall accuracy.
    overall_accuracy = (correct_overall / total_questions) * 100 if total_questions > 0 else 0
    print("\n=== Final Accuracy ===")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    print("\nAccuracy by Type:")
    for key, value in accuracy_by_type.items():
        acc = (value["correct"] / value["total"]) * 100 if value["total"] > 0 else 0
        print(f"  {key}: {acc:.2f}% ({value['correct']}/{value['total']})")

    print("\nAccuracy by Subtype:")
    for key, value in accuracy_by_subtype.items():
        acc = (value["correct"] / value["total"]) * 100 if value["total"] > 0 else 0
        print(f"  {key}: {acc:.2f}% ({value['correct']}/{value['total']})")

    print("\nAccuracy by Radius:")
    for key, value in accuracy_by_radius.items():
        acc = (value["correct"] / value["total"]) * 100 if value["total"] > 0 else 0
        print(f"  {key}: {acc:.2f}% ({value['correct']}/{value['total']})")


if __name__ == "__main__":
    main()
