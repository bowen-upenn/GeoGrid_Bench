import ast
import re
import pandas as pd
import argparse


climate_variables = {'maximum annual temperature': './data/climrr/AnnualTemperatureMaximum.csv',
                     'minimum annual temperature': './data/climrr/AnnualTemperatureMinimum.csv',
                     'consecutive days with no precipitation': './data/climrr/ConsecutiveDayswithNoPrecipitation.csv',
                     'cooling degree days': './data/climrr/CoolingDegreeDays.csv',
                     'fire weather index': './data/climrr/FireWeatherIndex_Wildfire.csv',
                     'maximum daily heat index': './data/climrr/heatindex.csv',
                     'maximum seasonal heat index': './data/climrr/heatindex.csv',
                     'number of days with daily heat index above 95 F': './data/climrr/heatindex.csv',
                     'number of days with daily heat index above 105 F': './data/climrr/heatindex.csv',
                     'number of days with daily heat index above 115 F': './data/climrr/heatindex.csv',
                     'number of days with daily heat index above 125 F': './data/climrr/heatindex.csv',
                     'heating degree': './data/climrr/HeatingDegreeDays.csv',
                     'annual total precipitation': './data/climrr/Precipitation_inches_AnnualTotal.csv',
                     'maximum seasonal temperature': './data/climrr/SeasonalTemperatureMaximum.csv',
                     'minimum seasonal temperature': './data/climrr/SeasonalTemperatureMinimum.csv',
                     'wind speed': './data/climrr/WindSpeed.csv'}


class Colors:
    HEADER = '\033[95m'  # Purple
    OKBLUE = '\033[94m'  # Blue
    OKGREEN = '\033[92m'  # Green
    WARNING = '\033[93m'  # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'     # Reset color


def all_climrr_datasets():
    return ['AnnualTemperatureMaximum.csv',
            'AnnualTemperatureMinimum.csv',
            'ConsecutiveDayswithNoPrecipitation.csv',
            'CoolingDegreeDays.csv',
            'FireWeatherIndex_Wildfire.csv',
            'heatindex.csv',
            'HeatingDegreeDays.csv',
            'Precipitation_inches_AnnualTotal.csv',
            'SeasonalTemperatureMaximum.csv',
            'SeasonalTemperatureMinimum.csv',
            'WindSpeed.csv']


def parse_inputs(args):
    city = args['datasets']['city']
    time = args['datasets']['time']
    dataset = args['datasets']['dataset']

    # check if each of the variable above is a single entity or a list of entities
    list_pattern = r'\[.*?,.*?\]'
    input_strings = [city, time, dataset]
    for i, input_string in enumerate(input_strings):
        if re.search(list_pattern, input_string):
            # Parse the string into a list using ast.literal_eval to safely evaluate the string
            try:
                entity_list = ast.literal_eval(input_string)
                if isinstance(entity_list, list):
                    input_strings[i] = entity_list
                else:
                    return "Invalid list format"
            except Exception as e:
                return f"Error parsing list: {e}"
        else:
            input_strings[i] = [input_string]

    city, time, dataset = input_strings[0], input_strings[1], input_strings[2]
    return city, time, dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--dataset', type=str, default="FireWeatherIndex_Wildfire", help='Name of the dataset')
    cmd_args = parser.parse_args()

    data_path = './data/climrr/' + cmd_args.dataset + '.csv'
    print('Looking at all possible time frames of the dataset' + data_path)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)
    # Print all column names
    print("Column names:", df.columns.tolist())