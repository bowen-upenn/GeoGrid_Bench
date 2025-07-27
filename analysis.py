import pandas as pd
import json
import re
import numpy as np
from scipy import stats # For ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd # For Post-Hoc test

# --- Configuration ---
QA_DATA_PATH = 'data/benchmark/qa_data.csv' # Contains "Question ID" and "Question" (filled string)
FILLED_QUESTIONS_PATH = 'data/all_filled_questions.json' # Contains filled_values, keyed by template
RESULT_PATH = 'result/eval_results_o4-mini_text.jsonl' # Model's results file

print("Analyzing Results from:", RESULT_PATH)

# Your provided list of all possible cities, NOW IN "City, ST" format.
ALL_POSSIBLE_CITIES = [
    "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ", "Philadelphia, PA",
    "San Antonio, TX", "San Diego, CA", "Dallas, TX", "Tampa, FL", "Austin, TX", "Fort Worth, TX",
    "San Jose, CA", "Columbus, OH", "Charlotte, NC", "Indianapolis, IN", "San Francisco, CA",
    "Seattle, WA", "Denver, CO", "Oklahoma City, OK", "Nashville, TN", "Washington, DC",
    "El Paso, TX", "Las Vegas, NV", "Boston, MA", "Detroit, MI", "Portland, OR",
    "Louisville, KY", "Baltimore, MD", "Milwaukee, WI", "Albuquerque, NM", "Tucson, AZ",
    "Fresno, CA", "Sacramento, CA", "Mesa, AZ", "Atlanta, GA", "Kansas City, MO",
    "Colorado Springs, CO", "Omaha, NE", "Raleigh, NC", "Miami, FL", "Virginia Beach, VA",
    "Long Beach, CA", "Oakland, CA", "Minneapolis, MN", "Bakersfield, CA", "Arlington, TX",
    "Wichita, KS", "Aurora, CO", "New Orleans, LA", "Cleveland, OH", "Honolulu, HI",
    "Anaheim, CA", "Henderson, NV", "Orlando, FL", "Lexington, KY", "Stockton, CA",
    "Riverside, CA", "Irvine, CA", "Cincinnati, OH", "Santa Ana, CA", "Newark, NJ",
    "Saint Paul, MN", "Pittsburgh, PA", "Greensboro, NC", "Durham, NC", "Lincoln, NE",
    "Jersey City, NJ", "Plano, TX", "North Las Vegas, NV", "St. Louis, MO", "Madison, WI",
    "Chandler, AZ", "Gilbert, AZ", "Reno, NV", "Buffalo, NY", "Chula Vista, CA",
    "Fort Wayne, IN", "Lubbock, TX", "Laredo, TX", "Toledo, OH", "Chesapeake, VA", "Glendale, AZ",
    "Winston-Salem, NC", "Port St. Lucie, FL", "Garland, TX", "Spokane, WA", "Richmond, VA",
    "Fremont, CA", "Salt Lake City, UT", "Yonkers, NY", "Worcester, MA", "Rochester, NY",
    "Columbus, GA", "Santa Rosa, CA", "Kansas City, KS", "Sunnyvale, CA", "Bellevue, WA",
    "Urbana, IL", "Syracuse, NY", "Charleston, SC"
]
ALL_POSSIBLE_CITIES_LOWER_SET = {city_st.lower() for city_st in ALL_POSSIBLE_CITIES}


# Curated list of typically "major" US cities, NOW IN "City, ST" format.
MAJOR_CITIES = [
    'New York, NY', 'Los Angeles, CA', 'Chicago, IL', 'Houston, TX', 'Phoenix, AZ',
    'Philadelphia, PA', 'San Antonio, TX', 'San Diego, CA', 'Dallas, TX', 'Austin, TX',
    'Washington, DC', 'Boston, MA', 'Seattle, WA', 'Miami, FL', 'Atlanta, GA',
    'San Francisco, CA', 'Denver, CO', 'Las Vegas, NV', 'Portland, OR', 'New Orleans, LA'
]
MAJOR_CITIES_LOWER_SET = {city_st.lower() for city_st in MAJOR_CITIES}


# Geographic Mapping (US Census Regions)
state_to_region = {
    'AL': 'South', 'AK': 'West', 'AZ': 'West', 'AR': 'South', 'CA': 'West', 'CO': 'West',
    'CT': 'Northeast', 'DE': 'South', 'DC': 'South', 'FL': 'South', 'GA': 'South', 'HI': 'West',
    'ID': 'West', 'IL': 'Midwest', 'IN': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'KY': 'South',
    'LA': 'South', 'ME': 'Northeast', 'MD': 'South', 'MA': 'Northeast', 'MI': 'Midwest', 'MN': 'Midwest',
    'MS': 'South', 'MO': 'Midwest', 'MT': 'West', 'NE': 'Midwest', 'NV': 'West', 'NH': 'Northeast',
    'NJ': 'Northeast', 'NM': 'West', 'NY': 'Northeast', 'NC': 'South', 'ND': 'Midwest', 'OH': 'Midwest',
    'OK': 'South', 'OR': 'West', 'PA': 'Northeast', 'RI': 'Northeast', 'SC': 'South', 'SD': 'Midwest',
    'TN': 'South', 'TX': 'South', 'UT': 'West', 'VT': 'Northeast', 'VA': 'South', 'WA': 'West',
    'WV': 'South', 'WI': 'Midwest', 'WY': 'West'
}

# --- 1. Load All Data Sources ---

# Load qa_data.csv - CRITICAL CHANGE HERE: Use "Question ID" and "Question"
try:
    df_qa_data = pd.read_csv(QA_DATA_PATH)
    # Check for required columns using their EXACT names
    required_qa_cols_exact = ["Question ID", "Filled Template Question"] # Exact names from error
    if not all(col in df_qa_data.columns for col in required_qa_cols_exact):
        print(f"Error: '{QA_DATA_PATH}' must contain columns: {required_qa_cols_exact}. Found: {df_qa_data.columns.tolist()}")
        exit()
    # Standardize 'Question ID' to 'question_id' for consistent merging
    df_qa_data['question_id'] = df_qa_data["Question ID"].astype(str)
    # Standardize 'Question' to 'question' (lowercase) for consistent merging with filled_questions
    df_qa_data['question_'] = df_qa_data["Question"].astype(str)
    df_qa_data['question'] = df_qa_data["Filled Template Question"]
except FileNotFoundError:
    print(f"Error: The QA data file '{QA_DATA_PATH}' was not found.")
    exit()
except Exception as e:
    print(f"Error loading '{QA_DATA_PATH}': {e}")
    exit()

# Load all_filled_questions.json
filled_questions_list = []
try:
    with open(FILLED_QUESTIONS_PATH, 'r') as f:
        template_data = json.load(f)
        for template_key, questions_for_template in template_data.items():
            for q_obj in questions_for_template:
                q_obj['question_template'] = template_key # Add template key
                # Ensure 'question' and 'filled_values' are present in each object
                if 'question' in q_obj and 'filled_values' in q_obj:
                    filled_questions_list.append(q_obj)
                else:
                    print(f"Warning: Skipping malformed question object in '{FILLED_QUESTIONS_PATH}': Missing 'question' or 'filled_values'. Object: {q_obj}")
    df_filled_questions = pd.DataFrame(filled_questions_list)
    if df_filled_questions.empty:
        print(f"Error: No valid question objects with 'question' and 'filled_values' found in '{FILLED_QUESTIONS_PATH}'. Check file content.")
        exit()
except FileNotFoundError:
    print(f"Error: The filled questions file '{FILLED_QUESTIONS_PATH}' was not found.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from '{FILLED_QUESTIONS_PATH}'. {e}")
    print("Please ensure 'all_filled_questions.json' is a valid JSON file (single object).")
    exit()
except Exception as e:
    print(f"Error processing '{FILLED_QUESTIONS_PATH}': {e}")
    exit()

# Load model results (eval_results_InternVL2_5-8B_text.jsonl)
model_results_raw = []
try:
    with open(RESULT_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                model_results_raw.append(json.loads(line))
except FileNotFoundError:
    print(f"Error: The model results file '{RESULT_PATH}' was not found.")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON from a line in '{RESULT_PATH}'. {e}")
    print("Please ensure each line in the file is a valid JSON object (JSONL format).")
    exit()
except Exception as e:
    print(f"Error loading '{RESULT_PATH}': {e}")
    exit()
df_results = pd.DataFrame(model_results_raw)


# --- Merge DataFrames ---
# 1. Filter model results to only 'question' type entries and ensure question_id is str
df_results_questions = df_results[df_results['entry_type'] == 'question'].copy()
required_result_cols = ['question_id', 'is_correct']
if not all(col in df_results_questions.columns for col in required_result_cols):
    required_result_cols = ['question', 'is_correct']
    df_results_questions['question_id'] = df_results_questions['question']
df_results_questions['question_id'] = df_results_questions['question_id'].astype(str) # Ensure string type

# IMPORTANT CHANGE: Ensure df_filled_questions has unique 'question' before merging
# If there are duplicate 'question' strings in df_filled_questions, decide how to handle them.
# For simplicity, we'll take the first occurrence if duplicates exist, assuming they are consistent.
# If they are not consistent, you might need a more sophisticated de-duplication strategy.
df_filled_questions_unique = df_filled_questions.drop_duplicates(subset=['question'], keep='first')
if len(df_filled_questions) != len(df_filled_questions_unique):
    print(f"Warning: Removed {len(df_filled_questions) - len(df_filled_questions_unique)} duplicate 'question' entries from df_filled_questions to ensure 1:1 merge.")

# 2. Merge df_qa_data with the now de-duplicated df_filled_questions on 'question' string
# This merge is still potentially many-to-one if df_qa_data has duplicate 'question' for different 'question_id's.
df_base_info_merged = pd.merge(
    df_qa_data,
    df_filled_questions_unique, # Use the de-duplicated version
    on='question',
    how='inner',
    suffixes=('_qa', '_filled')
)

if df_base_info_merged.empty:
    print("Error: No matching 'question' entries found between QA data and filled questions data for base merge.")
    print("Please ensure the 'question' strings are identical in both files for merging.")
    print(f"Sample 'question' from {QA_DATA_PATH}:\n{df_qa_data['question'].head().to_string()}")
    print(f"Sample 'question' from {FILLED_QUESTIONS_PATH}:\n{df_filled_questions['question'].head().to_string()}")
    exit()

# At this point, df_base_info_merged might still have more rows than unique question_ids if a question_id
# appears multiple times in df_qa_data with the same 'question' but perhaps different metadata (unlikely based on your description),
# or if multiple question_ids map to the *same* question string.
# The critical step is that the final merge with df_results_questions should be on 'question_id'.

# 3. Final merge: Combine the base info (with filled_values) and model results using 'question_id'
# This merge will ensure that each row corresponds to a unique question_id from the results.

if required_result_cols == ['question_id', 'is_correct']:
    df_final_merged = pd.merge(
        df_base_info_merged,
        df_results_questions[['question_id', 'is_correct']], # Select only necessary columns from results
        on='question_id',
        how='inner', # Only analyze questions for which we have all base data and model results
        suffixes=('_base_data', '_model_result')
    )
else:
    df_results_questions['question_id'] = df_results_questions['question'].astype(str) # Ensure string type
    df_base_info_merged['question_id'] = df_base_info_merged['question_'].astype(str) # Ensure string type
    df_final_merged = pd.merge(
        df_base_info_merged,
        df_results_questions[['question_id', 'is_correct']], # Select only necessary columns from results
        on='question_id',
        how='inner', # Only analyze questions for which we have all base data and model results
        suffixes=('_base_data', '_model_result')
    )

if df_final_merged.empty:
    print("Error: No matching 'question_id' entries found after merging all data sources.")
    print("This means there's no overlap between questions in your QA data/filled questions and model results.")
    print("Please check consistency of 'question_id' columns between QA data and model results.")
    print(f"Sample question_ids from merged base data:\n{df_base_info_merged['question_id'].head().to_string()}")
    print(f"Sample question_ids from model results:\n{df_results_questions['question_id'].head().to_string()}")
    exit()

# After the final merge, ensure the DataFrame is unique by 'question_id'.
# If df_base_info_merged had duplicate 'question_id's (e.g., due to 'question' string duplication in df_qa_data
# that we didn't address), this line ensures we only keep one entry per question_id.
# However, given your output "Unique 'question_id': 800" for df_qa_data,
# it's more likely the duplicates were introduced from df_filled_questions having
# non-unique 'question' values that match multiple 'question_id's from df_qa_data.
df = df_final_merged.drop_duplicates(subset=['question_id'], keep='first').copy() # Ensure uniqueness by question_id

# print(f"--- Data Load and Merging Summary ---")
# print(f"Entries from {QA_DATA_PATH}: {len(df_qa_data)}")
# print(f"Entries from {FILLED_QUESTIONS_PATH}: {len(df_filled_questions)}")
# print(f"Entries from {RESULT_PATH} (filtered to 'question' type): {len(df_results_questions)}")
# print(f"Total questions successfully merged for analysis: {len(df)}")
# print("-" * 70)

# print("\n--- Duplicate Check and Merge Explanation ---")
# print(f"df_qa_data: {len(df_qa_data)} rows")
# print(f"  Unique 'Question ID': {df_qa_data['question_id'].nunique()}")
# print(f"  Unique 'Question': {df_qa_data['question'].nunique()}")

# print(f"\ndf_filled_questions: {len(df_filled_questions)} rows")
# print(f"  Unique 'question': {df_filled_questions['question'].nunique()}")
# print(f"  Unique 'question' (after de-duplication for merge): {df_filled_questions_unique['question'].nunique()}")

# print(f"\ndf_results_questions: {len(df_results_questions)} rows")
# print(f"  Unique 'question_id': {df_results_questions['question_id'].nunique()}")

# print(f"\nAfter merging df_qa_data and df_filled_questions_unique (df_base_info_merged): {len(df_base_info_merged)} rows")
# print(f"  Unique 'question_id': {df_base_info_merged['question_id'].nunique()}")
# print(f"  Unique 'question': {df_base_info_merged['question'].nunique()}")

# print(f"\nAfter final merge with df_results_questions (df_final_merged before final de-dupe): {len(df_final_merged)} rows")
# print(f"  Unique 'question_id': {df_final_merged['question_id'].nunique()}")
# print(f"  Unique 'question_template': {df_final_merged['question_template'].nunique()}")
# print(f"  Unique 'question': {df_final_merged['question'].nunique()}")

# print(f"\nFinal DataFrame 'df' (after final de-duplication by 'question_id'): {len(df)} rows")
# print(f"  Unique 'question_id': {df['question_id'].nunique()}")
# print(f"  Unique 'question_template': {df['question_template'].nunique()}")
# print(f"  Unique 'question': {df['question'].nunique()}") # Note: 'question' might not be unique if different question_ids share the same question string
# print("-" * 70)

# --- 2. Enhance Data with Geographic and Temporal Metadata (Using filled_values) ---

def parse_and_validate_locations(filled_values):
    """
    Extracts, validates, and counts unique 'City, ST' locations from filled_values.
    Returns (city_name, state_abbr, city_st_full) for the FIRST valid location if only ONE is found,
    else returns (None, None, None).
    """
    if not isinstance(filled_values, dict):
        return None, None, None

    valid_locations_parsed = []
    
    # Iterate through potential location keys (e.g., 'location1', 'location2', etc.)
    # We sort keys to ensure consistent order if multiple valid locations are found (e.g., location1 before location2)
    location_keys = sorted([k for k in filled_values if k.startswith('location') and filled_values[k] is not None])

    for key in location_keys:
        location_str_full = filled_values[key]
        if not isinstance(location_str_full, str):
            continue

        # Pattern for "City, ST, United States" or "City, ST"
        match = re.search(r'([A-Za-z\s-]+),\s*([A-Z]{2})(?:,\s*United States)?\b', location_str_full)
        if match:
            city_name = match.group(1).strip()
            state_abbr = match.group(2).strip()
            city_st_full = f"{city_name}, {state_abbr}"

            if city_st_full.lower() in ALL_POSSIBLE_CITIES_LOWER_SET:
                valid_locations_parsed.append((city_name, state_abbr, city_st_full))
    
    # Remove duplicates if the same location is listed under different keys (e.g., location1="LA, CA", location2="LA, CA")
    unique_valid_locations = list(set(valid_locations_parsed))

    if len(unique_valid_locations) == 1:
        return unique_valid_locations[0] # Return the single valid location tuple (city_name, state_abbr, city_st_full)
    else:
        # If 0 or >1 valid locations, return None for all, indicating exclusion
        return None, None, None


def get_region(state_abbr):
    return state_to_region.get(state_abbr, 'Unknown')

def is_major_city(city_st_full):
    return city_st_full.lower() in MAJOR_CITIES_LOWER_SET if city_st_full else False

def parse_and_validate_temporal_info(filled_values):
    """
    Extracts and validates a single temporal period from filled_values.
    Returns the classified temporal period string if exactly ONE is found,
    else returns None.
    """
    if not isinstance(filled_values, dict):
        return None

    valid_temporal_periods = set()
    
    temporal_keys = sorted([k for k in filled_values if k.startswith('time_frame') and filled_values[k] is not None])

    for key in temporal_keys:
        time_frame_str = filled_values[key]
        if not isinstance(time_frame_str, str):
            continue

        time_frame_lower = time_frame_str.lower()
        
        if 'end of the century' in time_frame_lower or 'end-century' in time_frame_lower or '2090s' in time_frame_lower or '2100' in time_frame_lower:
            valid_temporal_periods.add('End-Century (Future)')
        elif 'mid-century' in time_frame_lower or 'middle of the century' in time_frame_lower or '2050s' in time_frame_lower or '2060s' in time_frame_lower:
            valid_temporal_periods.add('Mid-Century (Future/Near)')
        elif 'historical' in time_frame_lower or 'past' in time_frame_lower or 'pre-industrial' in time_frame_lower or '1900s' in time_frame_lower or '2000s' in time_frame_lower:
            valid_temporal_periods.add('Historical (Past)')
        # Removed 'Unspecified/Other' as a valid *single* period. If it's the only one, it's still "None".
        # This function should only return a *specific* period, or None if ambiguous/unclassified.

    if len(valid_temporal_periods) == 1:
        return list(valid_temporal_periods)[0]
    else:
        # If 0 or >1 distinct temporal periods, return None
        return None


# Apply extraction functions using 'filled_values'
df[['extracted_city', 'extracted_state_abbr', 'extracted_city_st_full']] = df['filled_values'].apply(
    lambda x: pd.Series(parse_and_validate_locations(x))
)

df['region'] = df['extracted_state_abbr'].apply(get_region)
df['is_major_city'] = df['extracted_city_st_full'].apply(is_major_city) 

# Apply new temporal parsing
df['temporal_period'] = df['filled_values'].apply(parse_and_validate_temporal_info)

# Convert is_correct to numeric (0 for False, 1 for True) for easy calculation
df['is_correct_numeric'] = df['is_correct'].astype(int)


# Filter the DataFrame for geographic analysis based on successfully extracted single locations
# Questions with 0 or >1 valid locations will have extracted_city_st_full as None
df_geo_filtered = df[
    df['extracted_city_st_full'].notna() & # Only include rows where a single valid location was parsed
    (df['region'] != 'Unknown')            # Ensure the state mapped to a region
].copy()

# Filter the DataFrame for temporal analysis based on successfully extracted single temporal periods
df_temporal_filtered = df[
    df['temporal_period'].notna() # Only include rows where a single valid temporal period was parsed
].copy()


print("\n--- Geographic Analysis Filtering ---")
num_geo_excluded = len(df) - len(df_geo_filtered)
print(f"Questions excluded from geographic analysis (not single, identifiable 'City, ST' location OR unmapped state): {num_geo_excluded}")
print(f"Questions included in geographic analysis: {len(df_geo_filtered)}")
# print("Sample of processed data for geographic analysis (first 5 rows):\n")
# print(df_geo_filtered[['question_id', 'question', 'extracted_city_st_full', 'region', 'is_major_city', 'is_correct_numeric']].head().to_string(index=False))
print("-" * 70)


print("\n--- Temporal Analysis Filtering ---")
num_temporal_excluded = len(df) - len(df_temporal_filtered)
print(f"Questions excluded from temporal analysis (not single, identifiable temporal period): {num_temporal_excluded}")
print(f"Questions included in temporal analysis: {len(df_temporal_filtered)}")
# print("Sample of processed data for temporal analysis (first 5 rows):\n")
# print(df_temporal_filtered[['question_id', 'question', 'temporal_period', 'is_correct_numeric']].head().to_string(index=False))
print("-" * 70)


# --- 3. Calculate Performance Metrics and ANOVA ---

print("\n--- Model Performance Analysis ---")

# Overall Accuracy (using the combined filtered set that has both valid location and temporal info, if desired)
# For overall accuracy, it's perhaps best to use the broadest valid set that has at least *one* valid attribute.
# Or, keep it distinct for each analysis type as you've now filtered into df_geo_filtered and df_temporal_filtered.
# For consistency, let's report overall for the entire initial merged set (df), as a baseline.
overall_accuracy_initial = df['is_correct_numeric'].mean()
print(f"\nOverall Model Accuracy (across all initially merged questions): {overall_accuracy_initial:.2f}\n")


# --- ANOVA for Geographic Regions ---
print("\n--- Statistical Test: Accuracy by US Geographic Region ---")
# Use df_geo_filtered for geographic analysis
df_analysis_geo = df_geo_filtered.copy()
print(f"Number of questions used for Geographic Region analysis: {len(df_analysis_geo)}")
overall_geo_accuracy = df_analysis_geo['is_correct_numeric'].mean()
print(f"Overall Accuracy for Geographic Analysis Dataset: {overall_geo_accuracy:.2f}\n")


# Get groups for ANOVA
groups_regions_data = []
region_labels = []
for r in df_analysis_geo['region'].unique():
    data = df_analysis_geo['is_correct_numeric'][df_analysis_geo['region'] == r]
    if len(data) > 1: # Need at least 2 data points for a meaningful ANOVA group
        groups_regions_data.append(data)
        region_labels.append(r)

if len(groups_regions_data) > 1: # Need at least 2 groups to compare
    f_stat_region, p_val_region = stats.f_oneway(*groups_regions_data)
    print(f"ANOVA F-statistic: {f_stat_region:.3f}")
    print(f"P-value: {p_val_region:.3f}")

    if p_val_region < 0.05:
        print("Interpretation: There is a statistically significant difference in accuracy across geographic regions (p < 0.05).")
        print("\n--- Post-hoc Tukey HSD Test for Geographic Regions ---")
        tukey_result_region = pairwise_tukeyhsd(endog=df_analysis_geo['is_correct_numeric'],
                                                groups=df_analysis_geo['region'],
                                                alpha=0.05)
        print(tukey_result_region)
        print("\nInterpretation: Pairs with 'reject' = True show a statistically significant difference.")
    else:
        print("Interpretation: No statistically significant difference in accuracy across geographic regions (p >= 0.05).")
else:
    print("Not enough groups or data points to perform ANOVA for Geographic Regions (requires at least 2 groups with >1 data points each).")

accuracy_by_region = df_analysis_geo.groupby('region')['is_correct_numeric'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Num Questions'})
print("\nAccuracy by US Geographic Region Means:\n")
print(accuracy_by_region.to_markdown(floatfmt=".2f"))
print("-" * 70)


# --- Statistical Test for City Prominence ---
print("\n--- Statistical Test: Accuracy by City Prominence (Major vs. Other Cities) ---")
# Use df_geo_filtered for geographic analysis
print(f"Number of questions used for City Prominence analysis: {len(df_analysis_geo)}")
groups_prominence_data = []
major_city_data = df_analysis_geo['is_correct_numeric'][df_analysis_geo['is_major_city'] == True]
other_city_data = df_analysis_geo['is_correct_numeric'][df_analysis_geo['is_major_city'] == False]

# Check if both groups have enough data for comparison
if len(major_city_data) > 1 and len(other_city_data) > 1:
    f_stat_prominence, p_val_prominence = stats.f_oneway(major_city_data, other_city_data)
    print(f"ANOVA F-statistic: {f_stat_prominence:.3f}")
    print(f"P-value: {p_val_prominence:.3f}")

    if p_val_prominence < 0.05:
        print("Interpretation: There is a statistically significant difference in accuracy between Major and Other cities (p < 0.05).")
    else:
        print("Interpretation: No statistically significant difference in accuracy between Major and Other cities (p >= 0.05).")
else:
    print("Not enough data points to perform statistical test for City Prominence (requires at least 2 data points in each category).")


accuracy_by_city_prominence = df_analysis_geo.groupby('is_major_city')['is_correct_numeric'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Num Questions'})
accuracy_by_city_prominence.index = accuracy_by_city_prominence.index.map({True: 'Major City', False: 'Other City'})
print("\nAccuracy by City Prominence Means:\n")
print(accuracy_by_city_prominence.to_markdown(floatfmt=".2f"))
print("-" * 70)


# --- Temporal Analysis ---
print("\n--- Statistical Test: Accuracy by Temporal Period ---")
# Use df_temporal_filtered for temporal analysis
df_analysis_temporal = df_temporal_filtered.copy()
print(f"Number of questions used for Temporal Period analysis: {len(df_analysis_temporal)}")
overall_temporal_accuracy = df_analysis_temporal['is_correct_numeric'].mean()
print(f"Overall Accuracy for Temporal Analysis Dataset: {overall_temporal_accuracy:.2f}\n")


groups_temporal_data = []
temporal_labels = []
for t_period in df_analysis_temporal['temporal_period'].unique():
    data = df_analysis_temporal['is_correct_numeric'][df_analysis_temporal['temporal_period'] == t_period]
    if len(data) > 1:
        groups_temporal_data.append(data)
        temporal_labels.append(t_period)

if len(groups_temporal_data) > 1:
    f_stat_temporal, p_val_temporal = stats.f_oneway(*groups_temporal_data)
    print(f"ANOVA F-statistic: {f_stat_temporal:.3f}")
    print(f"P-value: {p_val_temporal:.3f}")

    if p_val_temporal < 0.05:
        print("Interpretation: There is a statistically significant difference in accuracy across temporal periods (p < 0.05).")
        print("\n--- Post-hoc Tukey HSD Test for Temporal Periods ---")
        tukey_result_temporal = pairwise_tukeyhsd(endog=df_analysis_temporal['is_correct_numeric'],
                                                  groups=df_analysis_temporal['temporal_period'],
                                                  alpha=0.05)
        print(tukey_result_temporal)
        print("\nInterpretation: Pairs with 'reject' = True show a statistically significant difference.")
    else:
        print("Interpretation: No statistically significant difference in accuracy across temporal periods (p >= 0.05).")
else:
    print("Not enough groups or data points to perform ANOVA for Temporal Periods (requires at least 2 groups with >1 data points each).")

accuracy_by_temporal_period = df_analysis_temporal.groupby('temporal_period')['is_correct_numeric'].agg(['mean', 'count']).rename(columns={'mean': 'Accuracy', 'count': 'Num Questions'})
print("\nAccuracy by Temporal Period Means:\n")
print(accuracy_by_temporal_period.to_markdown(floatfmt=".2f"))
print("-" * 70)


# --- Combined Analysis: Accuracy by Region and Temporal Period ---
print("\n--- Combined Analysis: Accuracy and Number of Questions by US Region and Temporal Period ---")

# For combined analysis, we need questions that have *both* a single identifiable region AND a single identifiable temporal period.
df_combined_analysis = df[
    df['extracted_city_st_full'].notna() &
    (df['region'] != 'Unknown') &
    df['temporal_period'].notna()
].copy()

num_combined_excluded = len(df) - len(df_combined_analysis)
print(f"Questions excluded from combined analysis (not single location AND/OR not single temporal period): {num_combined_excluded}")
print(f"Questions included in combined analysis: {len(df_combined_analysis)}")
print(f"Overall Accuracy for Combined Analysis Dataset: {df_combined_analysis['is_correct_numeric'].mean():.2f}\n")


if not df_combined_analysis.empty:
    regions_present = df_combined_analysis['region'].unique()
    temporal_periods_present = df_combined_analysis['temporal_period'].unique()
    multi_index = pd.MultiIndex.from_product([regions_present, temporal_periods_present], names=['region', 'temporal_period'])

    combined_accuracy_df = df_combined_analysis.groupby(['region', 'temporal_period'])['is_correct_numeric'].agg(['mean', 'count']).reindex(multi_index).fillna({'mean': np.nan, 'count': 0})

    # Unstack 'mean' and 'count' separately to create two sub-tables
    accuracy_table = combined_accuracy_df['mean'].unstack().rename_axis(columns=None)
    count_table = combined_accuracy_df['count'].unstack().rename_axis(columns=None)

    print("Accuracy by US Region and Temporal Period:\n")
    print(accuracy_table.to_markdown(floatfmt=".2f"))
    print("\nNumber of Questions by US Region and Temporal Period:\n")
    print(count_table.to_markdown(floatfmt=".0f")) # Format count as integer
else:
    print("No data available for combined analysis after filtering for single location and single temporal period.")
print("-" * 70)