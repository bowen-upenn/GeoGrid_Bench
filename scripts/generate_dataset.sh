#!/bin/bash

# Define the specific radius values
radius_values=(48 72 108)

# Loop through each radius value
for radius in "${radius_values[@]}"; do
    echo "Running with radius=$radius km"
    python generate_dataset.py --llm gpt-4o-mini --data test_filled_questions.json --radius "$radius"
done

echo "All executions completed."
