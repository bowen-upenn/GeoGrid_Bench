import os
import numpy as np
import torch
import json

from data_retrieval import retrieve_data_from_location
from query_llm import QueryLLM
import utils


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
    for i, data_sample in enumerate(dataloader('./data/all_filled_questions.json')):
        question = data_sample['question']
        filled_values = data_sample['filled_values']
        template = data_sample['template']

        # Load data for climate_variable1
        climate_variable = filled_values['climate_variable1']
        location_description = filled_values['location1']
        time_period = filled_values['time_frame1']
        data_var1, crossmodel_indices1 = retrieve_data_from_location(climate_variable, location_description, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
        data_var1 = utils.reformat_to_2d_table(data_var1, crossmodel_indices1)

        # Load data for climate_variable2
        data_var2 = None
        if 'climate_variable2' in filled_values:
            climate_variable = filled_values['climate_variable2']
            location_description = filled_values['location2'] if 'location2' in filled_values else 'location1'
            time_period = filled_values['time_frame2'] if 'time_frame2' in filled_values else 'time_frame1'
            data_var2, crossmodel_indices2 = retrieve_data_from_location(climate_variable, location_description, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
            data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
        else:
            if 'location2' in filled_values:
                # Same climate variable but at two different locations
                assert 'time_frame2' not in filled_values
                location_description = filled_values['location2']
                data_var2, crossmodel_indices2 = retrieve_data_from_location(climate_variable, location_description, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)
            elif 'time_frame2' in filled_values:
                # Same climate variable but at two different time periods
                assert 'location2' not in filled_values
                time_period = filled_values['time_frame2']
                data_var2, crossmodel_indices2 = retrieve_data_from_location(climate_variable, location_description, time_period, llm, args['inference']['geometry'], args['inference']['radius'])
                data_var2 = utils.reformat_to_2d_table(data_var2, crossmodel_indices2)

        if args['inference']['verbose']:
            print(f'{utils.Colors.OKGREEN}Question:{utils.Colors.ENDC}')
            print(question)
            print(f'{utils.Colors.OKGREEN}Filled values:{utils.Colors.ENDC}')
            print(filled_values)

            print(f'{utils.Colors.OKGREEN}Data 1:{utils.Colors.ENDC}')
            print(data_var1)
            if data_var2 is not None:
                print(f'{utils.Colors.OKGREEN}Data 2:{utils.Colors.ENDC}')
                print(data_var2)
