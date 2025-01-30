import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from esda.moran import Moran
from libpysal.weights import lat2W
import random

import ocr
from templates import visual_qa_mode
import utils


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

    # if abs(moran.I) > 0.3:
    #     spatial_variation = "large spatial variations"
    # elif abs(moran.I) > 0.1:
    #     spatial_variation = "moderate spatial variations"
    # else:
    #     spatial_variation = "little spatial variations"
    # random_spacial_variation = np.random.choice(["large spatial variations", "moderate spatial variations", "little spatial variations"], 1)[0]  # could still be the correct one
    #
    # if verbose:
    #     print("Moran's I", moran.I)

    # return spatial_variation, random_spacial_variation
    return abs(moran.I)


def extract_place_names(ppocr, llm, template, correct_region, angle1, overlay1, overlay_path1, location_description1, data_df2=None, overlay2=None, overlay_path2=None, location_description2=None, verbose=False):
    ocr_results, names_in_regions, invalid_names = ppocr.run_ocr_detection(overlay_path1, angle1)

    correct_answer_place_names2, incorrect_answers_place_names2 = None, None
    if visual_qa_mode[template] is None:
        return None, None
    elif visual_qa_mode[template] == 'region':
        random_incorrect_region = np.random.choice([region for region in names_in_regions.keys() if region != correct_region])
        correct_answer_place_names, incorrect_answers_place_names = ocr.randomly_sample_place_names(names_in_regions[correct_region], names_in_regions[random_incorrect_region], invalid_names)
        correct_answer_place_names = utils.filter_names(llm, correct_answer_place_names, location_description1)
        incorrect_answers_place_names = [utils.filter_names(llm, names, location_description1) for names in incorrect_answers_place_names]

    elif visual_qa_mode[template] == 'block' or visual_qa_mode[template] == 'block2':
        red_set, blue_set = ocr.classify_bounding_boxes_by_color(llm, overlay1, ocr_results, location_description1)

        # find the intersection between names in the correct region and names in the red_set
        print('red_set', red_set, "correct_region", correct_region, "names_in_regions", names_in_regions)
        red_set = utils.match_tolerant_sets(red_set, names_in_regions[correct_region])
        print('matched red set', red_set)
        # red_set = list(set(names_in_regions[correct_region]) & set(red_set))

        correct_answer_place_names, incorrect_answers_place_names = ocr.randomly_sample_place_names(red_set, blue_set, invalid_names)
        if visual_qa_mode[template] == 'block2':
            ocr_results2, names_in_regions2 = ppocr.run_ocr_detection(overlay_path2, angle2)
            red_set2, blue_set2 = ocr.classify_bounding_boxes_by_color(llm, overlay2, ocr_results2, location_description2)
            correct_answer_place_names2, incorrect_answers_place_names2 = ocr.randomly_sample_place_names(red_set2, blue_set2, invalid_names)
    else:
        raise ValueError(f"Error: The visual QA mode {visual_qa_mode[template]} is not supported.")

    if len(correct_answer_place_names) == 0 or len(incorrect_answers_place_names) == 0:
        return None, None

    if verbose:
        print('correct_answer_place_names', correct_answer_place_names, 'incorrect_answers_place_names', incorrect_answers_place_names)

    try:
        if len(correct_answer_place_names) == 2:
            correct_answer_place_names = ' and '.join(correct_answer_place_names)
        else:
            correct_answer_place_names = ', '.join(correct_answer_place_names[:-1]) + ', and ' + correct_answer_place_names[-1] if len(correct_answer_place_names) > 1 else correct_answer_place_names[0]
        for i, names in enumerate(incorrect_answers_place_names):
            if len(names) == 2:
                names = ' and '.join(names)
            else:
                names = ', '.join(names[:-1]) + ', and ' + names[-1] if len(names) > 1 else names[0]
            incorrect_answers_place_names[i] = names

        if correct_answer_place_names2 is not None:
            if len(correct_answer_place_names2) == 2:
                correct_answer_place_names2 = ' and '.join(correct_answer_place_names2)
            else:
                correct_answer_place_names2 = ', '.join(correct_answer_place_names2[:-1]) + ', and ' + correct_answer_place_names2[-1] if len(correct_answer_place_names2) > 1 else correct_answer_place_names2[0]
            for i, names in enumerate(incorrect_answers_place_names2):
                if len(names) == 2:
                    names = ' and '.join(names)
                else:
                    names = ', '.join(names[:-1]) + ', and ' + names[-1] if len(names) > 1 else names[0]
                incorrect_answers_place_names2[i] = names
            return correct_answer_place_names, incorrect_answers_place_names, correct_answer_place_names2, incorrect_answers_place_names2

        return correct_answer_place_names, incorrect_answers_place_names
    except Exception as e:
        return None, None


def sample_row_col_indices_from_region(regions, target_region, k=3, incorrect=False):
    correct_region = regions[target_region]
    if incorrect:
        # Randomly sample a region different from the target region
        incorrect_region = np.random.choice([region for region in regions.keys() if region != target_region])
        incorrect_region = regions[incorrect_region]
        random_k_values = incorrect_region.unstack().dropna().sample(n=min(len(incorrect_region), k))
        sampled_values = random_k_values
        # all_values = correct_region.unstack().dropna()
        # top_k_values = all_values.nlargest(k)
        # remaining_values = all_values[~all_values.index.isin(top_k_values.index)]
        # if len(remaining_values) >= k:
        #     sampled_values = remaining_values.sample(n=k, random_state=42)
        # else:
        #     sampled_values = remaining_values  # Fallback in case there are fewer remaining values than k
    else:
        top_k_values = correct_region.unstack().dropna().nlargest(min(len(correct_region), k))    # Always select the grid with the largest value within the target region
        print('target_region', target_region, 'top_k_values', top_k_values)
        sampled_values = top_k_values

    sampled_indices = [f"C{sampled_values.index[i][0]} R{sampled_values.index[i][1]}" for i in range(k)]
    if len(sampled_indices) == 2:
        sampled_indices = ' and '.join(sampled_indices)
    else:
        sampled_indices = ', '.join(sampled_indices[:-1]) + ', and ' + sampled_indices[-1] if len(sampled_indices) > 1 else sampled_indices[0]
    return sampled_indices


def oracle_codes(ppocr, llm, template, data_var1, angle1, overlay1, overlay_path1, location_description1, data_var2=None, data_var3=None, data_var4=None, angle2=None, overlay2=None, overlay_path2=None, location_description2=None, verbose=False):
    if template == "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?":
        """
        The oracle code first divides the data into 9 regions and calculates the average value for each region.
        """
        # Flip the matrix upside down to match the map orientation
        data_var1 = data_var1.iloc[::-1]
        regions = divide_into_regions(data_var1)

        region_averages = {}
        for name, region_df in regions.items():
            # Calculate the average value for each region
            if region_df.size == 0 or region_df.isna().sum().sum() > region_df.size / 2:
                region_averages[name] = -np.inf
            else:
                region_averages[name] = region_df.mean().mean()

        # Identify the region with the highest average value
        max_region = max(region_averages, key=lambda k: (region_averages[k] if not np.isnan(region_averages[k]) else -np.inf))
        print('max_region', max_region)
        other_regions = [region for region in region_averages.keys() if region != max_region]
        other_regions = np.random.choice(other_regions, 3)
        correct_place_names, incorrect_place_names = extract_place_names(ppocr, llm, template, max_region, angle1, overlay1, overlay_path1, location_description1, verbose=verbose)

        # Prepare answers
        top_k_indices_correct = sample_row_col_indices_from_region(regions, max_region, k=2, incorrect=False)
        top_k_indices_incorrect = [sample_row_col_indices_from_region(regions, region, k=2, incorrect=True) for region in other_regions]

        correct_answer = {
            'location': {
                'region': f"The region around {max_region} experienced the largest increase.",
                'indices': f"The region around blocks {top_k_indices_correct} experienced the largest increase.",
                'places': f"The region around the textual marks {correct_place_names} on the map experienced the largest increase." if correct_place_names is not None else None,
            },
        }
        incorrect_answers = {
            'location': {
                'region': [
                    f"The region around {other_regions[0]} experienced the largest increase.",
                    f"The region around {other_regions[1]} experienced the largest increase.",
                    f"The region around {other_regions[2]} experienced the largest increase.",
                ],
                'indices': [
                    f"The region around blocks {top_k_indices_incorrect[0]} experienced the largest increase.",
                    f"The region around blocks {top_k_indices_incorrect[1]} experienced the largest increase.",
                    f"The region around blocks {top_k_indices_incorrect[2]} experienced the largest increase.",
                ],
                'places': [
                    f"The region around the textual marks {incorrect_place_names[0]} on the map experienced the largest increase.",
                    f"The region around the textual marks {incorrect_place_names[1]} on the map experienced the largest increase.",
                    f"The region around the textual marks {incorrect_place_names[2]} on the map experienced the largest increase.",
                ] if incorrect_place_names is not None else None,
            },
        }
        return correct_answer, incorrect_answers


    elif template == "What is the spatial variation of {climate_variable1} in {location1} during {time_frame1}?":
        """
        This oracle code calculates the spatial variation of the data and identifies the region with the largest spatial variation.
        """
        # Flip the matrix upside down to match the map orientation
        data_var1 = data_var1.iloc[::-1]
        regions = divide_into_regions(data_var1)

        # Identify the region with the highest spatial variation
        spatial_var = [calculate_spatial_variations(region_df) if region_df.isna().sum().sum() < region_df.size / 2 else -np.inf for region_df in regions.values()]
        max_region_var = list(regions.keys())[np.argmax(spatial_var)]
        other_regions_var = [region for region in regions.keys() if region != max_region_var]
        other_regions_var = np.random.choice(other_regions_var, 3)
        correct_place_names_var, incorrect_place_names_var = extract_place_names(ppocr, llm, template, max_region_var, angle1, overlay1, overlay_path1, location_description1, verbose=verbose)

        # Prepare answers
        top_k_indices_correct_var = sample_row_col_indices_from_region(regions, max_region_var, k=2, incorrect=False)
        top_k_indices_incorrect_var = [sample_row_col_indices_from_region(regions, region, k=2, incorrect=True) for region in other_regions_var]

        correct_answer = {
            'variation': {
                'region': f"The region around {max_region_var} experienced the largest spatial variation.",
                'indices': f"The region around blocks {top_k_indices_correct_var} experienced the largest spatial variation.",
                'places': f"The region around the textual marks {correct_place_names_var} on the map experienced the largest spatial variation." if correct_place_names_var is not None else None,
            },
        }
        incorrect_answers = {
            'variation': {
                'region': [
                    f"The region around {other_regions_var[0]} experienced the largest spatial variation.",
                    f"The region around {other_regions_var[1]} experienced the largest spatial variation.",
                    f"The region around {other_regions_var[2]} experienced the largest spatial variation.",
                ],
                'indices': [
                    f"The region around blocks {top_k_indices_incorrect_var[0]} experienced the largest spatial variation.",
                    f"The region around blocks {top_k_indices_incorrect_var[1]} experienced the largest spatial variation.",
                    f"The region around blocks {top_k_indices_incorrect_var[2]} experienced the largest spatial variation.",
                ],
                'places': [
                    f"The region around the textual marks {incorrect_place_names_var[0]} on the map experienced the largest spatial variation.",
                    f"The region around the textual marks {incorrect_place_names_var[1]} on the map experienced the largest spatial variation.",
                    f"The region around the textual marks {incorrect_place_names_var[2]} on the map experienced the largest spatial variation.",
                ] if incorrect_place_names_var is not None else None,
            },
        }
        return correct_answer, incorrect_answers


    elif template == "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?":
        """
        This oracle code calculates the change in a climate variable between two time frames in a given location.
        Based on the difference map, it calculates the change for each of the 9 regions and determine the change in each region.
        """
        # Flip the matrix upside down to match the map orientation
        data_var1 = data_var1.iloc[::-1]
        data_var2 = data_var2.iloc[::-1]

        # Divide the data into regions
        regions_var1 = divide_into_regions(data_var1)
        regions_var2 = divide_into_regions(data_var2)

        diff_table = data_var1 - data_var2
        regions_diff = divide_into_regions(diff_table)

        # Analyze each region
        region_changes = []
        region_changes_percent = []
        for region_name in regions_var1.keys():
            region1 = regions_var1[region_name].values.flatten()
            region2 = regions_var2[region_name].values.flatten()

            # Mask invalid (NaN) values
            valid_mask = ~np.isnan(region1) & ~np.isnan(region2)
            if np.any(valid_mask):  # Ensure there's valid data
                mean1 = np.mean(region1[valid_mask])
                mean2 = np.mean(region2[valid_mask])

                if regions_diff[region_name].isna().sum().sum() > regions_diff[region_name].size / 2:
                    change = -np.inf
                    percent = -np.inf
                else:
                    change = mean2 - mean1
                    percent = change / (mean1 + 1e-6)

                region_changes.append((region_name, change))
                region_changes_percent.append((region_name, percent))

        # Analyze overall trend
        max_region = max(region_changes, key=lambda x: abs(x[1]))[0]
        other_regions = [region for region, _ in region_changes if region != max_region]
        other_regions = np.random.choice(other_regions, 3)
        correct_place_names, incorrect_place_names = extract_place_names(ppocr, llm, template, max_region, angle1, overlay1, overlay_path1, location_description1, verbose=verbose)

        # Prepare answers
        top_k_indices_correct = sample_row_col_indices_from_region(regions_diff, max_region, k=2, incorrect=False)
        top_k_indices_incorrect = [sample_row_col_indices_from_region(regions_diff, region, k=2, incorrect=True) for region in other_regions]

        # Find the number of regions with increased and decreased values
        increase_count = sum(1 for _, change in region_changes if change > 0)
        decrease_count = sum(1 for _, change in region_changes if change < 0)

        # Find the overall percentage of increase or decrease relative to data_var1
        total_change = sum(change for _, change in region_changes)
        percent = total_change / np.nansum(data_var1)

        if np.abs(max([percent[1] for percent in region_changes_percent])) < 0.05:
            correct_trend = "no significant changes"
            incorrect_trend = ["most regions increased", "most regions decreased", "there are large variations across regions"]
        elif increase_count >= 8:     # with decrease_count <= 1 out of 9
            correct_trend = "most regions increased"
            incorrect_trend = ["most regions decreased", "there are large variations across regions", "no significant changes"]
        elif decrease_count >= 8:   # with increase_count <= 1 out of 9
            correct_trend = "most regions decreased"
            incorrect_trend = ["most regions increased", "there are large variations across regions", "no significant changes"]
        elif increase_count >= 6:   # with decrease_count <= 3 out of 9
            correct_trend = "more than half of the regions increased"
            incorrect_trend = ["more than half of the regions decreased", "most regions decreased", "no significant changes"]
        elif decrease_count >= 6:   # with increase_count <= 3 out of 9
            correct_trend = "more than half of the regions decreased"
            incorrect_trend = ["more than half of the regions increased", "most regions increased", "no significant changes"]
        else:   # with increase_count <= 5 and decrease_count <= 4 or increase_count <= 4 and decrease_count <= 5 out of 9
            correct_trend = "there are large variations across regions"
            incorrect_trend = ["most regions increased", "most regions decreased", "no significant changes"]

        if np.abs(percent) < 0.1:
            change_magnitude = "slightly"
        elif np.abs(percent < 0.5):
            change_magnitude = "moderately"
        else:
            change_magnitude = "significantly"

        correct_trend = f"{correct_trend} {change_magnitude}." if correct_trend != "no significant changes" else correct_trend
        incorrect_trend = [f"{trend} {change_magnitude}." if trend != "no significant changes" else trend for trend in incorrect_trend]

        correct_answer = {
            'trend': {
                'region': f"Overall {correct_trend}",
            }
        }
        if correct_trend != "no significant changes":
            correct_answer.update({
                'location': {
                    'region': f"The region around {max_region} has the largest change.",
                    'indices': f"The region around blocks {top_k_indices_correct} has the largest change.",
                    'places': f"The region around the textual marks {correct_place_names} on the map has the largest change." if correct_place_names is not None else None
                },
            })
            correct_answer['merge_two'] = {
                'region': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['region'][1:],
                'indices': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['indices'][1:],
                'places': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['places'][1:] if correct_place_names is not None else None
            }

        incorrect_answers = {
            'trend': {
                'region': [
                    f"Overall {incorrect_trend[0]}",
                    f"Overall {incorrect_trend[1]}",
                    f"Overall {incorrect_trend[2]}",
                ]
            }
        }
        if correct_trend != "no significant changes":
            incorrect_answers.update({
                'location': {
                    'region': [
                        f"The region around {other_regions[0]} has the largest change.",
                        f"The region around {other_regions[1]} has the largest change.",
                        f"The region around {other_regions[2]} has the largest change.",
                    ],
                    'indices': [
                        f"The region around blocks {top_k_indices_incorrect[0]} has the largest change.",
                        f"The region around blocks {top_k_indices_incorrect[1]} has the largest change.",
                        f"The region around blocks {top_k_indices_incorrect[2]} has the largest change.",
                    ],
                    'places': [
                        f"The region around the textual marks {incorrect_place_names[0]} on the map has the largest change.",
                        f"The region around the textual marks {incorrect_place_names[1]} on the map has the largest change.",
                        f"The region around the textual marks {incorrect_place_names[2]} on the map has the largest change.",
                    ] if incorrect_place_names is not None else None
                },
                'merge_two': {'region': [], 'indices': [], 'places': []}
            })

            for i in range(3):
                incorrect_answers['merge_two']['region'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers[random_two[1]]['region'][i][1:])
                incorrect_answers['merge_two']['indices'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers[random_two[1]]['indices'][i][1:])
                incorrect_answers['merge_two']['places'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers[random_two[1]]['places'][i][1:] if correct_place_names is not None else None)

        return correct_answer, incorrect_answers


    elif template == "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?":
        """
        This oracle code calculates the correlation between two climate variables in a given location.
        It first normalizes the data, computes the difference table, divides the data into regions, and calculates the correlation for each region.
        It identifies the overall trend, the region with the highest correlation, and the region with the highest spatial variation in correlations. 
        """
        # Flip the matrix upside down to match the map orientation
        data_var1 = data_var1.iloc[::-1]
        data_var2 = data_var2.iloc[::-1]

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

                if regions_diff[region_name].isna().sum().sum() > regions_diff[region_name].size / 2:
                    corr = -np.inf
                region_results.append({"region": region_name, "correlation": corr})

        # Analyze overall trend
        positive_count = sum(1 for r in region_results if r["correlation"] > 0)
        negative_count = sum(1 for r in region_results if r["correlation"] < 0)

        # Analyze detailed locations
        max_region = max(region_results, key=lambda x: abs(x["correlation"]))
        max_region_trend = "positive" if max_region['correlation'] > 0 else "negative"
        max_region_value = max_region['correlation']
        max_region = max_region["region"]
        other_regions = [r["region"] for r in region_results if r["region"] != max_region]
        other_regions = np.random.choice(other_regions, 3)
        correct_place_names, incorrect_place_names = extract_place_names(ppocr, llm, template, max_region, angle1, overlay1, overlay_path1, location_description1, verbose=verbose)

        # Prepare answers
        top_k_indices_correct = sample_row_col_indices_from_region(regions_diff, max_region, k=2, incorrect=False)
        top_k_indices_incorrect = [sample_row_col_indices_from_region(regions_diff, region, k=2, incorrect=True) for region in other_regions]

        if verbose:
            print('positive_count', positive_count, 'negative_count', negative_count, 'max_region_value', max_region_value)
        if max_region_value < 0.2:
            correct_trend = "no significant"
            incorrect_trend = ["highly positive", "slightly positive", "highly negative"]
        elif positive_count >= 8:  # with negative_count <= 1 out of 9
            correct_trend = "highly positive"
            incorrect_trend = ["highly negative", "slightly negative", "no significant"]
        elif negative_count >= 8:  # with positive_count <= 1 out of 9
            correct_trend = "highly negative"
            incorrect_trend = ["highly positive", "slightly positive", "no significant"]
        elif positive_count >= 6:   # with negative_count <= 3 out of 9
            correct_trend = "slightly positive"
            incorrect_trend = ["highly negative", "slightly negative", "no significant"]
        elif negative_count >= 6:   # with positive_count <= 3 out of 9
            correct_trend = "slightly negative"
            incorrect_trend = ["highly positive", "slightly positive", "no significant"]
        else:   # with positive_count <= 5 and negative_count <= 4 or positive_count <= 4 and negative_count <= 5 out of 9
            correct_trend = "no significant"
            incorrect_trend = ["highly positive", "slightly positive", "highly negative"]

        correct_answer = {
            'trend': {
                'region': f"Overall {correct_trend}.",
            }
        }
        if correct_trend != "no significant":
            correct_answer.update({
                'location': {
                    'region': f"The region around {max_region} has the largest {max_region_trend} correlation.",
                    'indices': f"The region around blocks {top_k_indices_correct} has the largest {max_region_trend} correlation.",
                    'places': f"The region around the textual marks {correct_place_names} on the map has the largest {max_region_trend} correlation." if correct_place_names is not None else None
                },
            })
            correct_answer['merge_two'] = {
                'region': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['region'][1:],
                'indices': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['indices'][1:],
                'places': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['places'][1:] if correct_place_names is not None else None
            }

        incorrect_answers = {
            'trend': {
                'region': [
                    f"Overall {incorrect_trend[0]} correlation.",
                    f"Overall {incorrect_trend[1]} correlation.",
                    f"Overall {incorrect_trend[2]} correlation.",
                ]
            }
        }
        if correct_trend != "no significant":
            incorrect_answers.update({
                'location': {
                    'region': [
                        f"The region around {other_regions[0]} has the largest {max_region_trend} correlation.",
                        f"The region around {other_regions[1]} has the largest {max_region_trend} correlation.",
                        f"The region around {other_regions[2]} has the largest {max_region_trend} correlation.",
                    ],
                    'indices': [
                        f"The region around blocks {top_k_indices_incorrect[0]} has the largest {max_region_trend} correlation.",
                        f"The region around blocks {top_k_indices_incorrect[1]} has the largest {max_region_trend} correlation.",
                        f"The region around blocks {top_k_indices_incorrect[2]} has the largest {max_region_trend} correlation.",
                    ],
                    'places': [
                        f"The region around the textual marks {incorrect_place_names[0]} on the map has the largest {max_region_trend} correlation.",
                        f"The region around the textual marks {incorrect_place_names[1]} on the map has the largest {max_region_trend} correlation.",
                        f"The region around the textual marks {incorrect_place_names[2]} on the map has the largest {max_region_trend} correlation.",
                    ] if incorrect_place_names is not None else None
                },
                'merge_two': {'region': [], 'indices': [], 'places': []}
            })
            for i in range(3):
                incorrect_answers['merge_two']['region'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['region'][i][1:])
                incorrect_answers['merge_two']['indices'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['indices'][i][1:])
                incorrect_answers['merge_two']['places'].append(incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['places'][i][1:] if correct_place_names is not None else None)

        return correct_answer, incorrect_answers


    elif template == "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?":
        """
        This oracle code calculates the seasonal variation of a single climate variable
        across four different data sets (e.g., representing four seasons or monthly subsets)
        in a given location and time frame.

        Steps:
        1. Flip each data_var (1–4) upside down to match the map orientation.
        2. Divide each of the four seasonal data sets into regions.
        3. For each region:
           - Compute the average value in each season.
           - Compute the standard deviation across the four seasonal averages (to capture
             how much the region varies from season to season).
        4. Use that standard deviation to judge whether seasonal variation is low, moderate,
           or high overall, and find which region has the highest variation.
        5. For that highest-variation region, additionally describe the seasonal trend
           (e.g., which season has the lowest average, which has the highest).
        6. Prepare correct and incorrect answers accordingly, referencing location indices
           and optional OCR-based place names on the map.
        """

        # Helper function to generate a short text describing seasonal trends
        def describe_seasonal_trend(season_means):
            """
            Given a list of four seasonal means corresponding to Winter, Spring, Summer, and Fall,
            return a short descriptive sentence about how the variable changes through the seasons.

            season_means: [mean_winter, mean_spring, mean_summer, mean_fall].
            """
            # Define season names in order
            seasons = ['spring', 'summer', 'autumn', 'winter']

            # Determine trend (increase or decrease) between consecutive seasons
            trends = []
            for i in range(3):  # Compare each season with the next one
                if season_means[i + 1] > season_means[i]:
                    trends.append("increases")
                elif season_means[i + 1] < season_means[i]:
                    trends.append("decreases")
                else:
                    trends.append("remains constant")

            # Merge consecutive identical trends into groups
            merged_trends = []
            merged_seasons = [seasons[0]]  # Start with Winter

            for i in range(1, len(seasons)):
                if trends[i - 1] == trends[i - 2] if i >= 2 else False:
                    merged_seasons[-1] += f" -> {seasons[i]}"  # Merge into previous group
                else:
                    merged_trends.append(trends[i - 1])
                    merged_seasons.append(seasons[i])

            # Construct trend description
            trend_description = ", then ".join(
                f"from {merged_seasons[i]} the variable {merged_trends[i]}"
                for i in range(len(merged_trends))
            ) + "."
            correct_trend_description = f"Throughout the year, {trend_description} "

            # Generate incorrect trend descriptions by modifying the correct trend
            incorrect_trends = []
            incorrect_trend_variations = [
                trends[::-1],  # Reverse the trends
                [random.choice(["increases", "decreases", "remains constant"]) for _ in trends],  # Random trends
                trends[:1] + trends[2:3] + [trends[1]]  # Swap middle trends
            ]

            for trend_variation in incorrect_trend_variations:
                merged_trends_alt = []
                merged_seasons_alt = [seasons[0]]

                for i in range(1, len(seasons)):
                    if trend_variation[i - 1] == trend_variation[i - 2] if i >= 2 else False:
                        merged_seasons_alt[-1] += f" → {seasons[i]}"  # Merge into previous group
                    else:
                        merged_trends_alt.append(trend_variation[i - 1])
                        merged_seasons_alt.append(seasons[i])

                trend_description_alt = ", then ".join(
                    f"from {merged_seasons_alt[i]} the variable {merged_trends_alt[i]}"
                    for i in range(len(merged_trends_alt))
                ) + "."

                incorrect_trends.append(trend_description_alt)

            # Return formatted description including seasonal means
            return correct_trend_description, incorrect_trends


        # 1. Flip matrices to match orientation
        data_var1 = data_var1.iloc[::-1]
        data_var2 = data_var2.iloc[::-1]
        data_var3 = data_var3.iloc[::-1]
        data_var4 = data_var4.iloc[::-1]

        # 2. Divide each of the four data sets into regions
        regions_season1 = divide_into_regions(data_var1)
        regions_season2 = divide_into_regions(data_var2)
        regions_season3 = divide_into_regions(data_var3)
        regions_season4 = divide_into_regions(data_var4)

        region_results = []

        # 3. For each region, compute seasonal means + standard deviation of those means
        all_region_names = set(regions_season1.keys())  # They should match in principle
        for region_name in all_region_names:
            region_data1 = regions_season1[region_name].values.flatten()
            region_data2 = regions_season2[region_name].values.flatten()
            region_data3 = regions_season3[region_name].values.flatten()
            region_data4 = regions_season4[region_name].values.flatten()

            # Validity checks: require enough non-NaN data in each season
            # to consider the region. You can adjust the threshold if desired.
            valid_counts = [
                np.sum(~np.isnan(region_data1)),
                np.sum(~np.isnan(region_data2)),
                np.sum(~np.isnan(region_data3)),
                np.sum(~np.isnan(region_data4)),
            ]
            total_pixels = len(region_data1)  # same length in each season's region
            # skip if fewer than 50% valid in *any* season
            if any(vc / total_pixels <= 0.5 for vc in valid_counts):
                continue

            # Compute average for each season ignoring NaNs
            mean_s1 = np.nanmean(region_data1)
            mean_s2 = np.nanmean(region_data2)
            mean_s3 = np.nanmean(region_data3)
            mean_s4 = np.nanmean(region_data4)

            season_means = [mean_s1, mean_s2, mean_s3, mean_s4]

            # Compute standard deviation across these four means
            stdev_across_seasons = np.nanstd(season_means)

            region_results.append({
                "region": region_name,
                "season_means": season_means,
                "avg_stdev": stdev_across_seasons
            })

        # If no valid regions, return an empty result
        if not region_results:
            return None, None

        # 4. Analyze overall magnitude of seasonal variation
        # Extract all stdev values
        all_stdevs = [res["avg_stdev"] for res in region_results]
        overall_mean_stdev = np.mean(all_stdevs)

        # You can define thresholds to label the overall seasonal variation
        if overall_mean_stdev < 0.3:
            variation_level = "Overall, there is low seasonal variation. "
            alt_variations = ["Overall, there is strong seasonal variation. ", "Overall, there is high seasonal variation. ", "Overall, there is almost no variation. "]
        elif overall_mean_stdev > 0.7:
            variation_level = "Overall, there is high seasonal variation. "
            alt_variations = ["Overall, there is low seasonal variation. ", "Overall, there is little seasonal variation. ", "Overall, there is almost no variation. "]
        else:
            variation_level = ""    # Omit this part if the variation is moderate to make the correct answer less ambiguous
            alt_variations = ["", "", ""]

        # 5. Find the region with the highest stdev
        max_region_entry = max(region_results, key=lambda x: x["avg_stdev"])
        max_region = max_region_entry["region"]
        max_value = max_region_entry["avg_stdev"]
        max_region_season_means = max_region_entry["season_means"]

        # Also pick 3 "other" regions for incorrect answers
        other_regions = [r for r in region_results if r["region"] != max_region]
        if len(other_regions) > 3:
            other_regions = np.random.choice(other_regions, 3, replace=False)

        # 6. Extract place names for the region with highest variation
        correct_place_names, incorrect_place_names = extract_place_names(
            ppocr, llm, template, max_region,
            angle1, overlay1, overlay_path1, location_description1,
            verbose=verbose
        )

        # Prepare sample indices
        stacked = np.dstack((
            data_var1.values,
            data_var2.values,
            data_var3.values,
            data_var4.values
        ))
        stdev_map = np.nanstd(stacked, axis=2)
        stdev_df = pd.DataFrame(stdev_map, index=data_var1.index, columns=data_var1.columns)
        regions_stdev_df = divide_into_regions(stdev_df)

        # Now sample
        top_k_indices_correct = sample_row_col_indices_from_region(regions_stdev_df, max_region, k=2, incorrect=False)
        top_k_indices_incorrect = []
        for reg_info in other_regions:
            reg_name = reg_info["region"]
            top_k_indices_incorrect.append(
                sample_row_col_indices_from_region(regions_stdev_df, reg_name, k=2, incorrect=True)
            )

        # Compute overall seasonal means across the entire map
        overall_mean_winter = np.nanmean(data_var1.values)
        overall_mean_spring = np.nanmean(data_var2.values)
        overall_mean_summer = np.nanmean(data_var3.values)
        overall_mean_fall = np.nanmean(data_var4.values)

        # Create a list of global seasonal means
        overall_season_means = [
            overall_mean_winter,
            overall_mean_spring,
            overall_mean_summer,
            overall_mean_fall
        ]

        # Compute the overall seasonal trend across the map
        overall_season_trend, incorrect_season_trends = describe_seasonal_trend(overall_season_means)

        # -- Prepare the correct answer --
        correct_answer = {
            'trend': {
                'region': f"{variation_level}{overall_season_trend}",
            },
            'location': {
                'region': f"The region around {max_region} has the largest seasonal variation ",
                'indices': f"The region around blocks {top_k_indices_correct} has the highest seasonal variation.",
                'places': f"The region around the textual marks {correct_place_names} on the map has the largest seasonal variation." if correct_place_names is not None else None
            }
        }
        correct_answer['merge_two'] = {
            'region': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['region'][1:],
            'indices': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['indices'][1:],
            'places': correct_answer['trend']['region'][:-1] + ' and t' + correct_answer['location']['places'][1:] if correct_place_names is not None else None
        }

        # -- Prepare the incorrect answers --
        incorrect_answers = {
            'trend': {
                'region': [
                    f"{alt_variations[0]}{incorrect_season_trends[0]}",
                    f"{alt_variations[1]}{incorrect_season_trends[1]}",
                    f"{alt_variations[2]}{incorrect_season_trends[2]}",
                ]
            },
            'location': {
                'region': [
                    f"The region around {other_regions[0]['region']} has the largest seasonal variation.",
                    f"The region around {other_regions[1]['region']} has the largest seasonal variation.",
                    f"The region around {other_regions[2]['region']} has the largest seasonal variation.",
                ],
                'indices': [
                    f"The region around blocks {top_k_indices_incorrect[0]} exhibits the highest seasonal variation."
                    if len(top_k_indices_incorrect) > 0 else "No data for region indices.",
                    f"The region around blocks {top_k_indices_incorrect[1]} exhibits the highest seasonal variation."
                    if len(top_k_indices_incorrect) > 1 else "No data for region indices.",
                    f"The region around blocks {top_k_indices_incorrect[2]} exhibits the highest seasonal variation."
                    if len(top_k_indices_incorrect) > 2 else "No data for region indices.",
                ],
                'places': (
                    [
                        f"The region around the textual marks {incorrect_place_names[0]} on the map has the largest seasonal variation."
                        if incorrect_place_names and len(incorrect_place_names) > 0 else None,
                        f"The region around the textual marks {incorrect_place_names[1]} on the map has the largest seasonal variation."
                        if incorrect_place_names and len(incorrect_place_names) > 1 else None,
                        f"The region around the textual marks {incorrect_place_names[2]} on the map has the largest seasonal variation."
                        if incorrect_place_names and len(incorrect_place_names) > 2 else None,
                    ]
                    if incorrect_place_names is not None else None
                ),
            },
            'merge_two': {'region': [], 'indices': [], 'places': []}
        }
        for i in range(3):
            region_merged = incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['region'][i][1:]
            indices_merged = incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['indices'][i][1:]
            if incorrect_answers['location']['places'] is not None and incorrect_answers['location']['places'][i] is not None:
                places_merged = incorrect_answers['trend']['region'][i][:-1] + ' and t' + incorrect_answers['location']['places'][i][1:]
            else:
                places_merged = None

            incorrect_answers['merge_two']['region'].append(region_merged)
            incorrect_answers['merge_two']['indices'].append(indices_merged)
            incorrect_answers['merge_two']['places'].append(places_merged)

        return correct_answer, incorrect_answers


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