import os
import numpy as np
import torch
import json
import geopandas as gpd
from tqdm import tqdm

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

    for i, data_sample in tqdm(enumerate(dataloader('./data/test_filled_questions.json'))):
        if i < 3:
            continue
        if i == 4:
            break

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
        data_var1, crossmodel_indices1, latlong1 = retrieve_data_from_location(climate_variable1, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
        data_var1 = utils.reformat_to_2d_table(data_var1, crossmodel_indices1)

        # Load data for climate_variable2
        data_var2, location_description2 = None, None
        if 'climate_variable2' in filled_values:
            climate_variable2 = filled_values['climate_variable2']
            if time_period1 not in utils.full_time_frames[climate_variable2]:
                continue
            data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable2, location_description1, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
            data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
        else:
            if 'location2' in filled_values:
                # Same climate variable but at two different locations
                assert 'time_frame2' not in filled_values
                location_description2 = filled_values['location2']
                data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable1, location_description2, time_period1, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
            elif 'time_frame2' in filled_values:
                # Same climate variable but at two different time periods
                assert 'location2' not in filled_values
                time_period2 = filled_values['time_frame2']
                data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable1, location_description1, time_period2, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)

        qa = {"question": question, "rephrased_question": rephrased_question, "filled_values": filled_values, "template": template, "data_var1": data_var1, "latlong1": latlong1}
        if data_var2 is not None:
            qa["data_var2"] = data_var2

        """
        The following answers come from one of the top place names shown on the actual map
        """
        if 'location2' in filled_values:
            title1, title2 = filled_values['location1'], filled_values['location2']
        elif 'time_frame2' in filled_values:
            title1, title2 = filled_values['time_frame1'], filled_values['time_frame2']
            title2 = filled_values['time_frame2']
        else:
            title1, title2 = filled_values['climate_variable1'], filled_values['climate_variable2'] if 'climate_variable2' in filled_values else None

        heatmap1, overlay1, overlay_path1, overlay_width1, overlay_height1 = visualization.visualize_grids(data_var1, title1, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'], output_path='heatmap1', verbose=args['inference']['verbose'])
        heatmap2, overlay2, overlay_path2 = None, None, None
        if 'climate_variable2' in filled_values:
            heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2 = visualization.visualize_grids(data_var2, title2, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])
        elif 'location2' in filled_values or 'time_frame2' in filled_values:
            heatmap2, overlay2, overlay_path2, overlay_width2, overlay_height2 = visualization.visualize_grids(data_var2, title2, center_lat=latlong2[0], center_lon=latlong2[1], size_km=args['inference']['radius'], output_path='heatmap2', verbose=args['inference']['verbose'])

        if heatmap2 is not None:
            heatmap_merged = utils.merge_two_figures(heatmap1, heatmap2)
            overlay_merged = utils.merge_two_figures(overlay1, overlay2)
            heatmap_merged.save('heatmap_merged.png')
            overlay_merged.save('heatmap_overlay_merged.png')
            if args['inference']['verbose']:
                print("Merged heatmap and overlay saved.")

        """ 
        The following answers come from one of the following relative locations: upper-left, upper-mid, upper-right, mid-left, center, mid-right, lower-left, lower-mid, lower-right
        """
        correct_answer, incorrect_answers = oracle.oracle_codes(ppocr, llm, template, data_var1, overlay1, overlay_path1, location_description1, data_var2, overlay2, overlay_path2, location_description2, args['inference']['verbose'])
        qa["correct_answer"] = correct_answer
        qa["incorrect_answers"] = incorrect_answers

        if args['inference']['verbose']:
            utils.print_qa(qa)
