import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from esda.moran import Moran
from libpysal.weights import lat2W


def divide_into_regions(data_var):
    # Define the 3x3 region splits:
    n_rows, n_cols = data_var.shape

    # Split rows and columns into three segments as evenly as possible.
    row_third = n_rows // 3
    col_third = n_cols // 3

    upper_row_slice = slice(0, row_third if row_third > 0 else 1)
    mid_row_slice = slice(row_third, 2 * row_third if 2 * row_third < n_rows else n_rows - 1)
    lower_row_slice = slice(2 * row_third, n_rows)

    upper_col_slice = slice(0, col_third if col_third > 0 else 1)
    mid_col_slice = slice(col_third, 2 * col_third if 2 * col_third < n_cols else n_cols - 1)
    lower_col_slice = slice(2 * col_third, n_cols)

    # Compute averages for each of the 9 regions:
    regions = {
        "upper-left": data_var.iloc[upper_row_slice, upper_col_slice],
        "upper-mid": data_var.iloc[upper_row_slice, mid_col_slice],
        "upper-right": data_var.iloc[upper_row_slice, lower_col_slice],
        "mid-left": data_var.iloc[mid_row_slice, upper_col_slice],
        "center": data_var.iloc[mid_row_slice, mid_col_slice],
        "mid-right": data_var.iloc[mid_row_slice, lower_col_slice],
        "lower-left": data_var.iloc[lower_row_slice, upper_col_slice],
        "lower-mid": data_var.iloc[lower_row_slice, mid_col_slice],
        "lower-right": data_var.iloc[lower_row_slice, lower_col_slice]
    }
    return regions


def fill_nan_with_nearest_neighbor(matrix):
    """
    Function to fill NaNs using nearest neighbor from a random direction
    This one is only used to calculate the Moran's I
    """
    nan_indices = np.argwhere(np.isnan(matrix))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for row, col in nan_indices:
        np.random.shuffle(directions)  # Randomize directions
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < matrix.shape[0] and 0 <= c < matrix.shape[1]:  # Boundary check
                if not np.isnan(matrix[r, c]):  # Valid neighbor
                    matrix[row, col] = matrix[r, c]
                    break  # Stop once a neighbor is found
    return matrix


def calculate_spatial_variations(data_var, verbose=False):
    # Moran's I
    flat_diff = data_var.values
    flat_diff = fill_nan_with_nearest_neighbor(flat_diff)
    w = lat2W(data_var.shape[0], data_var.shape[1])
    moran = Moran(flat_diff, w)

    if abs(moran.I) > 0.3:
        spatial_variation = "large spatial variations"
    elif abs(moran.I) > 0.1:
        spatial_variation = "moderate spatial variations"
    else:
        spatial_variation = "little spatial variations"
    random_spacial_variation = np.random.choice(["large spatial variations", "moderate spatial variations", "little spatial variations"], 1)[0]  # could still be the correct one

    if verbose:
        print("Moran's I", moran.I)

    return spatial_variation, random_spacial_variation


def sample_row_col_indices_from_region(regions, target_region, k=3, incorrect=False):
    correct_region = regions[target_region]
    if incorrect: # random k values in the region except the top k values in that region
        all_values = correct_region.unstack().dropna()
        top_k_values = all_values.nlargest(k)
        remaining_values = all_values[~all_values.index.isin(top_k_values.index)]
        if len(remaining_values) >= k:
            sampled_values = remaining_values.sample(n=k, random_state=42)  # Ensure reproducibility with random_state
        else:
            sampled_values = remaining_values  # Fallback in case there are fewer remaining values than k
    else:
        top_k_values = correct_region.unstack().dropna().nlargest(k)
        sampled_values = top_k_values

    sampled_indices = [f"C{sampled_values.index[i][0]} R{sampled_values.index[i][1]}" for i in range(k)]
    if len(sampled_indices) == 2:
        sampled_indices = ' and '.join(sampled_indices)
    else:
        sampled_indices = ', '.join(sampled_indices[:-1]) + ', and ' + sampled_indices[-1] if len(sampled_indices) > 1 else sampled_indices[0]
    return sampled_indices


def oracle_codes(template, data_var1, data_var2=None, verbose=False):
    if template == "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?":
        """
        We divide the data into 9 regions and calculate the average value for each region to determine the region with the highest average value.
        """
        regions = divide_into_regions(data_var1)

        region_averages = {}
        for name, region_df in regions.items():
            # Calculate the average value for each region
            if region_df.size == 0:
                region_averages[name] = np.nan
            else:
                region_averages[name] = region_df.mean().mean()

        # Identify the region with the highest average value
        # print("Region averages:", region_averages)
        max_region = max(region_averages, key=lambda k: (region_averages[k] if not np.isnan(region_averages[k]) else -np.inf))
        correct_answer = {'words': f'The region with the highest average value is: {max_region}', 'indices': ''}

        # Find the top 3 values in the region and their indices in data_var
        top_k_indices = sample_row_col_indices_from_region(regions, max_region, k=2, incorrect=False)
        correct_answer['indices'] = f'The region with the highest average value is around blocks: {top_k_indices}'

        # Find three random incorrect answers
        incorrect_answers = {'words': [], 'indices': []}
        for name, region_df in regions.items():
            if name != max_region:
                incorrect_answers['words'].append(f'The region with the highest average value is: {name}')
                top_k_indices = sample_row_col_indices_from_region(regions, name, k=2, incorrect=True)
                incorrect_answers['indices'].append(f'The region with the highest average value is around blocks: {top_k_indices}')

        return correct_answer, incorrect_answers


    elif template == "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?":
        """
        Based on the difference map, we calculate the correlation between the two variables in each of the 9 regions: 
        upper-left, upper-mid, upper-right, mid-left, center, mid-right, lower-left, lower-mid, lower-right.
        If the correlation is above 0.5 or lower than -0.5 in more than 6 regions, we say the variables are highly positively or negatively correlated.
        If the highest correlation is above 0.5 or lower than -0.5, we find the region with the highest correlation and report it.
        Otherwise, we say there is no significant correlation.
        We also calculate Moran's I to assess spatial correlation, and interpret the value as little, moderate, or large spatial variations.
        """
        # Normalize the data
        norm_var1 = (data_var1 - data_var1.mean()) / data_var1.std()
        norm_var2 = (data_var2 - data_var2.mean()) / data_var2.std()

        # Compute the difference table
        diff_table = norm_var2 - norm_var1

        # Divide data into regions
        regions_var1 = divide_into_regions(norm_var1)
        regions_var2 = divide_into_regions(norm_var2)
        regions_diff = divide_into_regions(diff_table)

        region_results = []
        for region_name in regions_var1.keys():
            region1 = regions_var1[region_name].values.flatten()
            region2 = regions_var2[region_name].values.flatten()

            # Ignore regions with less than 50% valid data
            valid_mask = ~np.isnan(region1) & ~np.isnan(region2)
            valid_data_count = np.sum(valid_mask)
            if valid_data_count / len(region1) <= 0.5 or valid_data_count / len(region2) <= 0.5:
                continue

            if np.any(valid_mask):  # Ensure there's valid data
                if len(region1[valid_mask]) < 2 or len(region2[valid_mask]) < 2:
                    continue
                corr, _ = pearsonr(region1[valid_mask], region2[valid_mask])
                region_results.append({"region": region_name, "correlation": corr})

        # Analyze correlations
        positive_count = sum(1 for r in region_results if r["correlation"] > 0.5)
        negative_count = sum(1 for r in region_results if r["correlation"] < -0.5)
        max_corr_region = max(region_results, key=lambda x: abs(x["correlation"]))
        other_regions = [r["region"] for r in region_results if r["region"] != max_corr_region["region"]]
        random_other_region = np.random.choice(other_regions, 1)[0]

        # Calculate spatial variations
        spatial_variation, random_spacial_variation = calculate_spatial_variations(diff_table, verbose=verbose)

        # Determine overall result
        random_positive_or_negative = np.random.choice(["positive", "negative"], 1)[0]
        if positive_count >= 6:
            correct_answer = {'words': f'Highly positively correlated with {spatial_variation}', 'indices': f'Highly positively correlated with {spatial_variation}'}
            top_k_indices = sample_row_col_indices_from_region(regions_diff, random_other_region, k=2, incorrect=True)
            incorrect_answers = {'words': [
                "Highly negatively correlated with " + random_spacial_variation,
                "No significant correlation.",
                f"The region {random_other_region} has the highest {random_positive_or_negative} correlation" + ' with ' + random_spacial_variation,
            ], 'indices': [
                "Highly negatively correlated with " + random_spacial_variation,
                "No significant correlation.",
                f"The region around blocks {top_k_indices} has the highest {random_positive_or_negative} correlation with " + random_spacial_variation,
            ]}
        elif negative_count >= 6:
            correct_answer = {'words': f'Highly negatively correlated with {spatial_variation}', 'indices': f'Highly negatively correlated with {spatial_variation}'}
            top_k_indices = sample_row_col_indices_from_region(regions_diff, random_other_region, k=2, incorrect=True)
            incorrect_answers = {'words': [
                "Highly positively correlated with " + random_spacial_variation,
                "No significant correlation.",
                f"The region {random_other_region} has the highest {random_positive_or_negative} correlation" + ' with ' + random_spacial_variation,
            ], 'indices': [
                "Highly positively correlated with " + random_spacial_variation,
                "No significant correlation.",
                f"The region around blocks {top_k_indices} has the highest {random_positive_or_negative} correlation with " + random_spacial_variation,
            ]}
        elif abs(max_corr_region["correlation"]) > 0.5:
            top_k_indices = sample_row_col_indices_from_region(regions_diff, max_corr_region["region"], k=2, incorrect=False)
            correct_answer = {'words': f"The region {max_corr_region['region']} has the highest {'positive' if max_corr_region['correlation'] > 0 else 'negative'} correlation" + ' with ' + spatial_variation,
                              'indices': f"The region around blocks {top_k_indices} has the highest {'positive' if max_corr_region['correlation'] > 0 else 'negative'} correlation" + ' with ' + spatial_variation}
            incorrect_answers = {'words': ["No significant correlation."], 'indices': ["No significant correlation."]}
            for _ in range(2):
                random_region = np.random.choice(other_regions, 1)[0]
                top_k_indices = sample_row_col_indices_from_region(regions_diff, random_region, k=2, incorrect=True)
                incorrect_answers['words'].append(f"The region {random_region} has the highest {np.random.choice(['positive', 'negative'], 1)[0]} correlation with"
                                                    + np.random.choice(["large spatial variations", "moderate spatial variations", "little spatial variations"], 1)[0])
                incorrect_answers['indices'].append(f"The region around blocks {top_k_indices} has the highest {np.random.choice(['positive', 'negative'], 1)[0]} correlation with"
                                                    + np.random.choice(["large spatial variations", "moderate spatial variations", "little spatial variations"], 1)[0])
        else:
            correct_answer = {'words': "No significant correlation.", 'indices': "No significant correlation."}
            top_k_indices = sample_row_col_indices_from_region(regions_diff, random_other_region, k=2, incorrect=True)
            incorrect_answers = {'words': [
                "Highly positively correlated with " + random_spacial_variation,
                "Highly negatively correlated with " + random_spacial_variation,
                f"The region {random_other_region} has the highest {random_positive_or_negative} correlation with " + random_spacial_variation,
            ], 'indices': [
                "Highly positively correlated with " + random_spacial_variation,
                "Highly negatively correlated with " + random_spacial_variation,
                f"The region around blocks {top_k_indices} has the highest {random_positive_or_negative} correlation with " + random_spacial_variation,
            ]}

        return correct_answer, incorrect_answers


    elif template == "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?":
        """
        Analyzes how a climate variable has changed between two time frames in a given location.
        Based on the difference map, we calculate the change for each of the 9 regions and determine the change in each region.
        We categorize the changes as 'slightly', 'somewhat', or 'significantly' based on the percent change.
        """
        # Divide the data into regions
        regions_var1 = divide_into_regions(data_var1)
        regions_var2 = divide_into_regions(data_var2)

        diff_table = data_var1 - data_var2

        # Analyze each region
        region_changes = []
        for region_name in regions_var1.keys():
            region1 = regions_var1[region_name].values.flatten()
            region2 = regions_var2[region_name].values.flatten()

            # Mask invalid (NaN) values
            valid_mask = ~np.isnan(region1) & ~np.isnan(region2)
            if np.any(valid_mask):  # Ensure there's valid data
                mean1 = np.mean(region1[valid_mask])
                mean2 = np.mean(region2[valid_mask])
                change = (mean2 - mean1) / max(abs(mean1), 1e-6)  # Percent change relative to mean1
                region_changes.append((region_name, change))
        if verbose:
            print("Region changes:", region_changes)

        # Categorize changes
        increase_count = 0
        decrease_count = 0
        insignificant_count = 0
        quantifiers = []

        for _, change in region_changes:
            if abs(change) < 0.2:
                quantifiers.append("slightly")
                insignificant_count += 1
            elif abs(change) < 0.7:
                quantifiers.append("somewhat")
            else:
                quantifiers.append("significantly")

            if change > 0:
                increase_count += 1
            elif change < 0:
                decrease_count += 1

        def describe_majority():
            if increase_count > decrease_count and increase_count > insignificant_count:
                return "Increased", quantifiers[increase_count - 1]
            elif decrease_count > increase_count and decrease_count > insignificant_count:
                return "Decreased", quantifiers[decrease_count - 1]
            else:
                return "No significant change", "over time"

        majority_change, change_magnitude = describe_majority()

        # Calculate spatial variations
        spatial_variation, random_spatial_variation = calculate_spatial_variations(diff_table, verbose=verbose)

        # Generate incorrect answers
        random_positive_or_negative = np.random.choice(["positive", "negative"], 1)[0]
        random_other_region = np.random.choice(list(regions_var1.keys()), 1)[0]

        if majority_change == "Increased":
            correct_answer = f"{majority_change} {change_magnitude} with {spatial_variation}."
            incorrect_answers = [
                f"Decreased significantly with {random_spatial_variation}.",
                f"No significant change over time.",
                f"The region {random_other_region} has the highest {random_positive_or_negative} change with {random_spatial_variation}.",
            ]
        elif majority_change == "Decreased":
            correct_answer = f"{majority_change} {change_magnitude} with {spatial_variation}."
            incorrect_answers = [
                f"Increased significantly with {random_spatial_variation}.",
                f"No significant change over time.",
                f"The region {random_other_region} has the highest {random_positive_or_negative} change with {random_spatial_variation}.",
            ]
        else:
            correct_answer = f"{majority_change} {change_magnitude} with {spatial_variation}."
            incorrect_answers = [
                f"Increased significantly with {random_spatial_variation}.",
                f"Decreased significantly with {random_spatial_variation}.",
                f"The region {random_other_region} has the highest {random_positive_or_negative} change with {random_spatial_variation}.",
            ]

        return correct_answer, incorrect_answers


    elif template == "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?":
        pass

    elif template == "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?":
        pass

    elif template == "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?":
        pass

    elif template == "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1}?":
        pass

    elif template == "How does the seasonal variation of {climate_variable1} in {location1} compare to that in {location2} for {time_frame1}?":
        pass
    else:
        raise ValueError(f"Unknown template: {template}")