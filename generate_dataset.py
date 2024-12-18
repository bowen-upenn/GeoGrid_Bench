import os
import numpy as np
import torch
import json
import geopandas as gpd

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

    for i, data_sample in enumerate(dataloader('./data/all_filled_questions.json')):
        if i == 1:
            continue

        question = data_sample['question']
        filled_values = data_sample['filled_values']
        template = data_sample['template']
        rephrased_question = llm.query_llm(step='rephrase_question', content=question, assistant=False, verbose=False)

        # Load data for climate_variable1
        climate_variable = filled_values['climate_variable1']
        location_description1 = filled_values['location1']
        time_period = filled_values['time_frame1']
        data_var1, crossmodel_indices1, latlong1 = retrieve_data_from_location(climate_variable, location_description1, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
        data_var1 = utils.reformat_to_2d_table(data_var1, crossmodel_indices1)

        # Load data for climate_variable2
        data_var2 = None
        if 'climate_variable2' in filled_values:
            climate_variable = filled_values['climate_variable2']
            location_description2 = filled_values['location2'] if 'location2' in filled_values else 'location1'
            time_period = filled_values['time_frame2'] if 'time_frame2' in filled_values else 'time_frame1'
            data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable, location_description2, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
            data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
        else:
            if 'location2' in filled_values:
                # Same climate variable but at two different locations
                assert 'time_frame2' not in filled_values
                location_description2 = filled_values['location2']
                data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable, location_description2, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
            elif 'time_frame2' in filled_values:
                # Same climate variable but at two different time periods
                assert 'location2' not in filled_values
                time_period = filled_values['time_frame2']
                data_var2, crossmodel_indices2, latlong2 = retrieve_data_from_location(climate_variable, location_description2, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)

        qa = {"question": question, "rephrased_question": rephrased_question, "filled_values": filled_values, "template": template, "data_var1": data_var1, "latlong1": latlong1}
        if data_var2 is not None:
            qa["data_var2"] = data_var2

        answers = {"relative_locations": {}, "place_names": {}}
        correct_answer_relative_locations, incorrect_answers_relative_locations = oracle.oracle_codes(template, data_var1)
        answers["relative_locations"] = {"correct_answer": correct_answer_relative_locations, "incorrect_answers": incorrect_answers_relative_locations}

        heatmap, overlay, overlay_path, overlay_width, overlay_height = visualization.visualize_grids(data_var1, center_lat=latlong1[0], center_lon=latlong1[1], size_km=args['inference']['radius'])

        ocr_results = ppocr.run_ocr_detection(overlay_path)
        red_set, blue_set = visualization.classify_bounding_boxes_by_color(llm, overlay, ocr_results, location_description1)
        correct_answer_place_names, incorrect_answers_place_names = utils.randomly_sample_place_names(red_set, blue_set)
        answers["place_names"] = {"correct_answer": correct_answer_place_names, "incorrect_answers": incorrect_answers_place_names}

        qa["answers"] = answers
        if args['inference']['verbose']:
            utils.print_qa(qa)
