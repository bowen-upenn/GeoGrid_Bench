import ast
import re


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