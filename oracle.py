import numpy as np
import pandas as pd


def oracle_codes(template, cropped_df, verbose=False):
    if template == "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?":
        # We assume rows = time steps, columns = spatial positions.
        # Define the 3x3 region splits:
        n_rows, n_cols = cropped_df.shape

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
            "upper-left":  cropped_df.iloc[upper_row_slice, upper_col_slice],
            "upper-mid":   cropped_df.iloc[upper_row_slice, mid_col_slice],
            "upper-right": cropped_df.iloc[upper_row_slice, lower_col_slice],
            "mid-left":    cropped_df.iloc[mid_row_slice, upper_col_slice],
            "center":      cropped_df.iloc[mid_row_slice, mid_col_slice],
            "mid-right":   cropped_df.iloc[mid_row_slice, lower_col_slice],
            "lower-left":  cropped_df.iloc[lower_row_slice, upper_col_slice],
            "lower-mid":   cropped_df.iloc[lower_row_slice, mid_col_slice],
            "lower-right": cropped_df.iloc[lower_row_slice, lower_col_slice]
        }

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
        correct_answer = f'The region with the highest average value is: {max_region}'

        # Find three random incorrect answers
        incorrect_answers = []
        for name, region_df in regions.items():
            if name != max_region:
                incorrect_answers.append(f'The region with the highest average value is: {name}')

        if verbose:
            print(f'{utils.Colors.OKGREEN}correct_answer{utils.Colors.ENDC}')
        return correct_answer, incorrect_answers
