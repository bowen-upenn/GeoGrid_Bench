import numpy as np
import torch
import json

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
import yaml
import json
import argparse
import sys

from data_retrieval import retrieve_data_from_location
from query_llm import QueryLLM
import utils
import visualization
import ocr
import oracle


def dataloader(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            for entry in data:
                yield entry
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode the JSON file {json_file_path}.")


def generate_dataset(args):
    llm = QueryLLM(args)
    ppocr = ocr.OCR()

    for i, data_sample in tqdm(enumerate(dataloader(args['inference']['data']))):
        # if i < 8:
        #     continue

        question_dir = f'output/question_{i}'
        os.makedirs(question_dir, exist_ok=True)

        question = data_sample['question']
        filled_values = data_sample['filled_values']
        template = data_sample['template']
        rephrased_question = llm.query_llm(step='rephrase_question', content=question, assistant=False, verbose=False)
        if args['inference']['verbose']:
            print(f'{utils.Colors.HEADER}Processing data sample {i}. Question {question}{utils.Colors.ENDC}')

        # Load data for climate_variable1
        climate_variable1 = filled_values['climate_variable1']
        location_description1 = filled_values['location1']
        time_period1 = filled_values['time_frame1']

        data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8 = None, None, None, None, None, None, None
        data_df2, data_df3, data_df4, data_df5, data_df6, data_df7, data_df8 = None, None, None, None, None, None, None
        heatmap2, overlay2, overlay_path2, data_df2, df_col_name2, cell_geometries2, angle2, heatmap3 = None, None, None, None, None, None, None, None

        if 'season' in template or 'seasonal' in template:
            # Initialize data variables for 4 seasons
            data_vars = {f"data_var{i}": None for i in range(1, 9)}
            data_dfs = {f"data_df{i}": None for i in range(1, 9)}
            crossmodel_indices = {f"crossmodel_indices{i}": None for i in range(1, 9)}
            df_col_names = {f"df_col_name{i}": None for i in range(1, 9)}
            location_description2 = None

            for i, season in enumerate(['spring', 'summer', 'autumn', 'winter', 'spring', 'summer', 'autumn', 'winter'], start=1):
                if i > 3 and 'location2' not in filled_values:
                    break
                key_var = f"data_var{i}"
                key_df = f"data_df{i}"
                key_crossmodel_indices = f"crossmodel_indices{i}"
                key_df_col_name = f"df_col_name{i}"

                # Retrieve data for each season. We name some of them as _1 consistently because they refer to the same location
                if i < 3:
                    data_vars[key_var], crossmodel_indices[key_crossmodel_indices], latlong1, data_dfs[key_df], df_col_names[key_df_col_name], cell_geometries1 = (
                        retrieve_data_from_location(climate_variable1, location_description1, f"{season} in {time_period1}", llm, args['inference']['geometry'], args['inference']['radius']))
                else: # Retrieve data for the second location
                    location_description2 = filled_values['location2']
                    data_vars[key_var], crossmodel_indices[key_crossmodel_indices], latlong2, data_dfs[key_df], df_col_names[key_df_col_name], cell_geometries2 = (
                        retrieve_data_from_location(climate_variable1, location_description2, f"{season} in {time_period1}", llm, args['inference']['geometry'], args['inference']['radius']))
                data_vars[key_var] = utils.reformat_to_2d_table(data_vars[key_var], crossmodel_indices[key_crossmodel_indices])

            data_var1, data_var2, data_var3, data_var4 = data_vars['data_var1'], data_vars['data_var2'], data_vars['data_var3'], data_vars['data_var4']
            data_df1, data_df2, data_df3, data_df4 = data_dfs['data_df1'], data_dfs['data_df2'], data_dfs['data_df3'], data_dfs['data_df4']
            df_col_name1, df_col_name2, df_col_name3, df_col_name4 = df_col_names['df_col_name1'], df_col_names['df_col_name2'], df_col_names['df_col_name3'], df_col_names['df_col_name4']
            if location_description2:
                data_var5, data_var6, data_var7, data_var8 = data_vars['data_var5'], data_vars['data_var6'], data_vars['data_var7'], data_vars['data_var8']
                data_df5, data_df6, data_df7, data_df8 = data_dfs['data_df5'], data_dfs['data_df6'], data_dfs['data_df7'], data_dfs['data_df8']
                df_col_name5, df_col_name6, df_col_name7, df_col_name8 = df_col_names['df_col_name5'], df_col_names['df_col_name6'], df_col_names['df_col_name7'], df_col_names['df_col_name8']
        else:
            data_var1, crossmodel_indices1, latlong1, data_df1, df_col_name1, cell_geometries1 = retrieve_data_from_location(climate_variable1, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
            data_var1 = utils.reformat_to_2d_table(data_var1, crossmodel_indices1)

            # Load data for climate_variable2
            location_description2 = None
            if 'climate_variable2' in filled_values:
                climate_variable2 = filled_values['climate_variable2']
                if time_period1 not in utils.full_time_frames[climate_variable2]:
                    continue
                data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable2, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
            else:
                if 'location2' in filled_values and 'time_frame2' not in filled_values:
                    # Same climate variable but at two different locations
                    assert 'time_frame2' not in filled_values
                    location_description2 = filled_values['location2']
                    data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description2, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
                    data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
                if 'time_frame2' in filled_values and 'location2' not in filled_values:
                    # Same climate variable but at two different time periods
                    assert 'location2' not in filled_values
                    time_period2 = filled_values['time_frame2']
                    data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description1, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
                    data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
                if 'time_frame2' in filled_values and 'location2' in filled_values:
                    location_description2 = filled_values['location2']
                    time_period2 = filled_values['time_frame2']
                    data_var2, crossmodel_indices2, latlong2, data_df2, df_col_name2, cell_geometries2 = retrieve_data_from_location(climate_variable1, location_description1, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
                    data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
                    data_var3, crossmodel_indices3, latlong3, data_df3, df_col_name3, cell_geometries3 = retrieve_data_from_location(climate_variable1, location_description2, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
                    data_var3 = utils.reformat_to_2d_table(data_var3, crossmodel_indices3)
                    data_var4, crossmodel_indices4, latlong4, data_df4, df_col_name4, cell_geometries4 = retrieve_data_from_location(climate_variable1, location_description2, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
                    data_var4 = utils.reformat_to_2d_table(data_var4, crossmodel_indices4)

        qa = {"question": question, "rephrased_question": rephrased_question, "filled_values": filled_values, "template": template, "data_var1": data_var1[::-1], "latlong1": latlong1}
        if data_var2 is not None:
            qa["data_var2"] = data_var2[::-1]
        if data_var3 is not None:
            qa["data_var3"] = data_var3[::-1]
        if data_var4 is not None:
            qa["data_var4"] = data_var4[::-1]

        """
        The following answers come from one of the top place names shown on the actual map
        """
        if 'season' in template or 'seasonal' in template:
            if 'location2' in filled_values:
                title1, title2, title3, title4 = 'Spring in ' + filled_values['location1'], 'Summer in ' + filled_values['location1'], 'Autumn in ' + filled_values['location1'], 'Winter in ' + filled_values['location1']
                title5, title6, title7, title8 = 'Spring in ' + filled_values['location2'], 'Summer in ' + filled_values['location2'], 'Autumn in ' + filled_values['location2'], 'Winter in ' + filled_values['location2']
            else:
                title1, title2, title3, title4 = 'Spring in ' + filled_values['time_frame1'], 'Summer in ' + filled_values['time_frame1'], 'Autumn in ' + filled_values['time_frame1'], 'Winter in ' + filled_values['time_frame1']
        elif 'location2' in filled_values and 'time_frame2' not in filled_values:
            title1, title2 = filled_values['location1'], filled_values['location2']
        elif 'time_frame2' in filled_values and 'location2' not in filled_values:
            title1, title2 = filled_values['time_frame1'], filled_values['time_frame2']
        elif 'location2' in filled_values and 'time_frame2' in filled_values:
            title1, title2, title3, title4 = filled_values['location1'] + ' in ' + filled_values['time_frame1'], filled_values['location1'] + ' in ' + filled_values['time_frame2'], filled_values['location2'] + ' in ' + filled_values['time_frame1'], filled_values['location2'] + ' in ' + filled_values['time_frame2']
        else:
            title1, title2 = filled_values['climate_variable1'], filled_values['climate_variable2'] if 'climate_variable2' in filled_values else None

        # Define a global minimum, maximum, and midpoint across both datasets
        valid_data_vars = [var for var in [data_var1, data_var2, data_var3, data_var4] if var is not None]
        global_min = np.nanmin([var.min().min() for var in valid_data_vars])
        global_max = np.nanmax([var.max().max() for var in valid_data_vars])
        global_midpoint = (global_max + global_min) / 2
        color_norm = TwoSlopeNorm(vmin=global_min, vmax=global_max, vcenter=global_midpoint)

        heatmap1, overlay1, overlay_path1, overlay_width1, overlay_height1, angle1 = visualization.visualize_grids(question_dir, data_df1, data_var1, title1, df_col_name1, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap1', verbose=args['inference']['verbose'])
        if ('season' in template or 'seasonal' in template) or ('location2' in filled_values and 'time_frame2' in filled_values):
            if 'location2' in filled_values and 'time_frame2' in filled_values:
                heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries1, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
                heatmap3, overlay3, overlay_path3, overlay_width3, overlay_height3, angle3 = visualization.visualize_grids(question_dir, data_df3, data_var3, title3, df_col_name3, cell_geometries2, color_norm, center_lat=latlong3[0], center_lon=latlong3[1], size_km=args['inference']['radius'], output_path='heatmap3', verbose=args['inference']['verbose'])
                heatmap4, overlay4, overlay_path4, overlay_width4, overlay_height4, angle4 = visualization.visualize_grids(question_dir, data_df4, data_var4, title4, df_col_name4, cell_geometries2, color_norm, center_lat=latlong4[0], center_lon=latlong4[1], size_km=args['inference']['radius'], output_path='heatmap4', verbose=args['inference']['verbose'])
            else:
                heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
                heatmap3, overlay3, overlay_path3, overlay_width3, overlay_height3, angle3 = visualization.visualize_grids(question_dir, data_df3, data_var3, title3, df_col_name3, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap3', verbose=args['inference']['verbose'])
                heatmap4, overlay4, overlay_path4, overlay_width4, overlay_height4, angle4 = visualization.visualize_grids(question_dir, data_df4, data_var4, title4, df_col_name4, cell_geometries1, color_norm, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap4', verbose=args['inference']['verbose'])

            if data_df5 is not None: # Seasons of the second location
                heatmap5, overlay5, overlay_path5, overlay_width5, overlay_height5, angle5 = visualization.visualize_grids(question_dir, data_df5, data_var5, title5, df_col_name5, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap5', verbose=args['inference']['verbose'])
                heatmap6, overlay6, overlay_path6, overlay_width6, overlay_height6, angle6 = visualization.visualize_grids(question_dir, data_df6, data_var6, title6, df_col_name6, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap6', verbose=args['inference']['verbose'])
                heatmap7, overlay7, overlay_path7, overlay_width7, overlay_height7, angle7 = visualization.visualize_grids(question_dir, data_df7, data_var7, title7, df_col_name7, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap7', verbose=args['inference']['verbose'])
                heatmap8, overlay8, overlay_path8, overlay_width8, overlay_height8, angle8 = visualization.visualize_grids(question_dir, data_df8, data_var8, title8, df_col_name8, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap8', verbose=args['inference']['verbose'])

        if ('climate_variable2' in filled_values) or ('location2' in filled_values and 'time_frame2' not in filled_values) or ('time_frame2' in filled_values and 'location2' not in filled_values):
            heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2, angle2 = visualization.visualize_grids(question_dir, data_df2, data_var2, title2, df_col_name2, cell_geometries2, color_norm, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])

        if heatmap3 is not None and heatmap4 is not None:
            heatmap_merged = utils.merge_two_figures(
                utils.merge_two_figures(heatmap1, heatmap2), utils.merge_two_figures(heatmap3, heatmap4), vertical=not ('season' in template or 'seasonal' in template))
            overlay_merged = utils.merge_two_figures(
                utils.merge_two_figures(overlay1, overlay2), utils.merge_two_figures(overlay3, overlay4), vertical=not ('season' in template or 'seasonal' in template))

            if heatmap5 is not None:
                heatmap_merged2 = utils.merge_two_figures(
                    utils.merge_two_figures(heatmap5, heatmap6), utils.merge_two_figures(heatmap7, heatmap8), vertical=False)
                overlay_merged2 = utils.merge_two_figures(
                    utils.merge_two_figures(overlay5, overlay6), utils.merge_two_figures(overlay7, overlay8), vertical=False)
                heatmap_merged = utils.merge_two_figures(heatmap_merged, heatmap_merged2, vertical=True)
                overlay_merged = utils.merge_two_figures(overlay_merged, overlay_merged2, vertical=True)

            heatmap_merged.save(os.path.join(question_dir, 'heatmap_merged.png'))
            overlay_merged.save(os.path.join(question_dir, 'heatmap_overlay_merged.png'))
        elif heatmap2 is not None:
            heatmap_merged = utils.merge_two_figures(heatmap1, heatmap2)
            overlay_merged = utils.merge_two_figures(overlay1, overlay2)
            heatmap_merged.save(os.path.join(question_dir, 'heatmap_merged.png'))
            overlay_merged.save(os.path.join(question_dir, 'heatmap_overlay_merged.png'))
        if args['inference']['verbose']:
            print("Merged heatmap and overlay saved.")

        """
        The following answers come from one of the following relative locations: upper-left, upper-mid, upper-right, mid-left, center, mid-right, lower-left, lower-mid, lower-right
        """
        correct_answer, incorrect_answers = oracle.oracle_codes(question_dir, ppocr, llm, template, data_var1, angle1, overlay1, overlay_path1, location_description1, location_description2,
                                                                angle2, overlay_path2, data_var2, data_var3, data_var4, data_var5, data_var6, data_var7, data_var8, verbose=args['inference']['verbose'])
        qa["correct_answer"] = correct_answer
        qa["incorrect_answers"] = incorrect_answers

        if args['inference']['verbose']:
            utils.print_qa(qa)


if __name__ == "__main__":
    print("Python", sys.version, 'Torch', torch.__version__)
    # Load hyperparameters
    try:
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--llm', type=str, default="gpt-4o-mini", help='Set LLM model. Choose from gpt-4-turbo, gpt-4o')
    parser.add_argument('--n', type=int, default=-1, help='Set number of samples to generate. Default is -1, which generates all samples')
    parser.add_argument('--data', type=str, default="test_filled_questions.json", help='Set data file to generate samples from')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()
    cmd_args.data = f'./data/{cmd_args.data}'

    # Override args from config.yaml with command-line arguments if provided
    args['models']['llm'] = cmd_args.llm if cmd_args.llm is not None else args['models']['llm']
    args['inference']['num_samples'] = cmd_args.n if cmd_args.n is not None else args['inference']['n']
    args['inference']['data'] = cmd_args.data if cmd_args.data is not None else args['inference']['data']
    args['inference']['verbose'] = cmd_args.verbose if cmd_args.verbose is not None else args['inference']['verbose']

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = torch.cuda.device_count()
    assert world_size == 1
    print('device', device)
    print('torch.distributed.is_available', torch.distributed.is_available())
    print('Using %d GPUs' % (torch.cuda.device_count()))

    # Start inference
    print(args)
    generate_dataset(args)