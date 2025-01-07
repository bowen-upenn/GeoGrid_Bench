import ast
import re
import pandas as pd
import argparse
import random
from PIL import Image
import json


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


allowed_time_frames = {'maximum annual temperature': {'historical period': 'hist',
                                                      'mid-century period (RCP4.5)': 'rcp45_midc',
                                                      'end-century period (RCP4.5)': 'rcp45_endc',
                                                      'mid-century period (RCP8.5)': 'rcp85_midc',
                                                      'end-century period (RCP8.5)': 'rcp85_endc'},

                        'minimum annual temperature': {'historical period': 'hist',
                                                       'mid-century period (RCP4.5)': 'rcp45_midc',
                                                       'end-century period (RCP4.5)': 'rcp45_endc',
                                                       'mid-century period (RCP8.5)': 'rcp85_midc',
                                                       'end-century period (RCP8.5)': 'rcp85_endc'},

                        'consecutive days with no precipitation': {'historical period': 'hist',
                                                                   'mid-century period (RCP4.5)': 'rcp45_midc',
                                                                   'end-century period (RCP4.5)': 'rcp45_endc',
                                                                   'mid-century period (RCP8.5)': 'rcp85_midc',
                                                                   'end-century period (RCP8.5)': 'rcp85_endc'},

                        'cooling degree days': {'historical period': 'hist',
                                                'mid-century period (RCP8.5)': 'rcp85_midc'},

                        'fire weather index': {'spring in historical period': 'wildfire_spring_Hist',
                                           'spring in mid-century period': 'wildfire_spring_Midc',
                                           'spring in end-century period': 'wildfire_spring_Endc',
                                           'summer in historical period': 'wildfire_summer_Hist',
                                           'summer in mid-century period': 'wildfire_summer_Midc',
                                           'summer in end-century period': 'wildfire_summer_Endc',
                                           'autumn in historical period': 'wildfire_autumn_Hist',
                                           'autumn in mid-century period': 'wildfire_autumn_Midc',
                                           'autumn in end-century period': 'wildfire_autumn_Endc',
                                           'winter in historical period': 'wildfire_winter_Hist',
                                           'winter in mid-century period': 'wildfire_winter_Midc',
                                           'winter in end-century period': 'wildfire_winter_Endc'},

                        'maximum daily heat index': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                     'summer in mid-century period': 'heatindex_M85_DayMax',
                                                     'summer in end-century period': 'heatindex_E85_DayMax',
                                                     'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                     'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},
                        'maximum seasonal heat index': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                        'summer in mid-century period': 'heatindex_M85_DayMax',
                                                        'summer in end-century period': 'heatindex_E85_DayMax',
                                                        'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                        'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},

                        'number of days with daily heat index above 95 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                            'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                            'summer in end-century period': 'heatindex_E85_DayMax'},
                        'number of days with daily heat index above 105 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                             'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                             'summer in end-century period': 'heatindex_E85_DayMax'},
                        'number of days with daily heat index above 115 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                             'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                             'summer in end-century period': 'heatindex_E85_DayMax'},
                        'number of days with daily heat index above 125 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                             'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                             'summer in end-century period': 'heatindex_E85_DayMax'},

                        'heating degree': {'historical period': 'hist',
                                           'mid-century period': 'rcp85_midc'},

                        'annual total precipitation': {'historical period': 'hist',
                                                       'mid-century period (RCP4.5)': 'rcp45_midc',
                                                       'end-century period (RCP4.5)': 'rcp45_endc',
                                                       'mid-century period (RCP8.5)': 'rcp85_midc',
                                                       'end-century period (RCP8.5)': 'rcp85_endc'},

                        'maximum seasonal temperature': {'spring in historical period': 'hist_spring',
                                                         'spring in mid-century period (RCP8.5)': 'rcp85_midc_spring',
                                                         'spring in end-century period (RCP8.5)': 'rcp85_endc_spring',
                                                         'summer in historical period': 'hist_summer',
                                                         'summer in mid-century period (RCP8.5)': 'rcp85_midc_summer',
                                                         'summer in end-century period (RCP8.5)': 'rcp85_endc_summer',
                                                         'autumn in historical period': 'hist_autumn',
                                                         'autumn in mid-century period (RCP8.5)': 'rcp85_midc_autumn',
                                                         'autumn in end-century period (RCP8.5)': 'rcp85_endc_autumn',
                                                         'winter in historical period': 'hist_winter',
                                                         'winter in mid-century period (RCP8.5)': 'rcp85_midc_winter',
                                                         'winter in end-century period (RCP8.5)': 'rcp85_endc_winter'},

                        'minimum seasonal temperature': {'spring in historical period': 'hist_spring',
                                                         'spring in mid-century period (RCP8.5)': 'rcp85_midc_spring',
                                                         'spring in end-century period (RCP8.5)': 'rcp85_endc_spring',
                                                         'summer in historical period': 'hist_summer',
                                                         'summer in mid-century period (RCP8.5)': 'rcp85_midc_summer',
                                                         'summer in end-century period (RCP8.5)': 'rcp85_endc_summer',
                                                         'autumn in historical period': 'hist_autumn',
                                                         'autumn in mid-century period (RCP8.5)': 'rcp85_midc_autumn',
                                                         'autumn in end-century period (RCP8.5)': 'rcp85_endc_autumn',
                                                         'winter in historical period': 'hist_winter',
                                                         'winter in mid-century period (RCP8.5)': 'rcp85_midc_winter',
                                                         'winter in end-century period (RCP8.5)': 'rcp85_endc_winter'},

                        'wind speed': {'historical period': 'hist',
                                       'mid-century period (RCP4.5)': 'rcp45_midc',
                                       'end-century period (RCP4.5)': 'rcp45_endc',
                                       'mid-century period (RCP8.5)': 'rcp85_midc',
                                       'end-century period (RCP8.5)': 'rcp85_endc'},
                        }


# the following dictionary contains data for changes and differences between time frames, which could be used as ground truth answers
full_time_frames = {'maximum annual temperature': {'historical period': 'hist',
                                                   'mid-century period (RCP4.5)': 'rcp45_midc',
                                                   'end-century period (RCP4.5)': 'rcp45_endc',
                                                   'mid-century period (RCP8.5)': 'rcp85_midc',
                                                   'end-century period (RCP8.5)': 'rcp85_endc',
                                                   'changes between historical and mid-century period (RCP4.5)': 'mid45_hist',
                                                   'changes between historical and end-century period (RCP4.5)': 'end45_hist',
                                                   'changes between historical and mid-century period (RCP8.5)': 'mid85_hist',
                                                   'changes between historical and end-century period (RCP8.5)': 'end85_hist',
                                                   'difference in mid-century periods (RCP4.5 and RCP8.5)': 'mid85_45',
                                                   'difference in end-century periods (RCP4.5 and RCP8.5)': 'end85_45'},

                 'minimum annual temperature': {'historical period': 'hist',
                                                'mid-century period (RCP4.5)': 'rcp45_midc',
                                                'end-century period (RCP4.5)': 'rcp45_endc',
                                                'mid-century period (RCP8.5)': 'rcp85_midc',
                                                'end-century period (RCP8.5)': 'rcp85_endc',
                                                'change between historical and mid-century period (RCP4.5)': 'mid45_hist',
                                                'changes between historical and end-century period (RCP4.5)': 'end45_hist',
                                                'changes between historical and mid-century period (RCP8.5)': 'mid85_hist',
                                                'changes between historical and end-century period (RCP8.5)': 'end85_hist',
                                                'difference in mid-century periods (RCP4.5 and RCP8.5)': 'mid85_45',
                                                'difference in end-century periods (RCP4.5 and RCP8.5)': 'end85_45'},

                'consecutive days with no precipitation': {'historical period': 'hist',
                                                           'mid-century period (RCP4.5)': 'rcp45_midc',
                                                           'end-century period (RCP4.5)': 'rcp45_endc',
                                                           'mid-century period (RCP8.5)': 'rcp85_midc',
                                                           'end-century period (RCP8.5)': 'rcp85_endc',
                                                           'change between historical and mid-century period (RCP4.5)': 'mid45_hist',
                                                           'changes between historical and end-century period (RCP4.5)': 'end45_hist',
                                                           'changes between historical and mid-century period (RCP8.5)': 'mid85_hist',
                                                           'changes between historical and end-century period (RCP8.5)': 'end85_hist',
                                                           'difference in mid-century periods (RCP4.5 and RCP8.5)': 'mid85_45',
                                                           'difference in end-century periods (RCP4.5 and RCP8.5)': 'end85_45'},

                'cooling degree days': {'historical period': 'hist',
                                        'mid-century period (RCP8.5)': 'rcp85_midc',
                                        'changes between historical and mid-century period (RCP8.5)': 'mid85_hist'},

                'fire weather index': {'spring in historical period': 'wildfire_spring_Hist',
                                       'spring in mid-century period': 'wildfire_spring_Midc',
                                       'spring in end-century period': 'wildfire_spring_Endc',
                                       'difference in spring between mid-century and historical periods': 'wildfire_spring_Dmid',
                                       'difference in spring between end-century and historical periods': 'wildfire_spring_Dend',
                                       'percent change in spring between mid-century and historical periods': 'wildfire_spring_Pmid',
                                       'percent change in spring between end-century and historical periods': 'wildfire_spring_Pend',
                                       'summer in historical period': 'wildfire_summer_Hist',
                                       'summer in mid-century period': 'wildfire_summer_Midc',
                                       'summer in end-century period': 'wildfire_summer_Endc',
                                       'difference in summer between mid-century and historical periods': 'wildfire_summer_Dmid',
                                       'difference in summer between end-century and historical periods': 'wildfire_summer_Dend',
                                       'percent change in summer between mid-century and historical periods': 'wildfire_summer_Pmid',
                                       'percent change in summer between end-century and historical periods': 'wildfire_summer_Pend',
                                       'autumn in historical period': 'wildfire_autumn_Hist',
                                       'autumn in mid-century period': 'wildfire_autumn_Midc',
                                       'autumn in end-century period': 'wildfire_autumn_Endc',
                                       'difference in autumn between mid-century and historical periods': 'wildfire_autumn_Dmid',
                                       'difference in autumn between end-century and historical periods': 'wildfire_autumn_Dend',
                                       'percent change in autumn between mid-century and historical periods': 'wildfire_autumn_Pmid',
                                       'percent change in autumn between end-century and historical periods': 'wildfire_autumn_Pend',
                                       'winter in historical period': 'wildfire_winter_Hist',
                                       'winter in mid-century period': 'wildfire_winter_Midc',
                                       'winter in end-century period': 'wildfire_winter_Endc',
                                       'difference in winter between mid-century and historical periods': 'wildfire_winter_Dmid',
                                       'difference in winter between end-century and historical periods': 'wildfire_winter_Dend',
                                       'percent change in winter between mid-century and historical periods': 'wildfire_winter_Pmid',
                                       'percent change in winter between end-century and historical periods': 'wildfire_winter_Pend'},

                'maximum daily heat index': {'summer in historical period': 'heatindex_HIS_DayMax',
                                             'summer in mid-century period': 'heatindex_M85_DayMax',
                                             'summer in end-century period': 'heatindex_E85_DayMax',
                                             'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                             'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},
                'maximum seasonal heat index': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                'summer in mid-century period': 'heatindex_M85_DayMax',
                                                'summer in end-century period': 'heatindex_E85_DayMax',
                                                'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},

                'number of days with daily heat index above 95 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                    'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                    'summer in end-century period': 'heatindex_E85_DayMax',
                                                                    'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                                    'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},
                'number of days with daily heat index above 105 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                     'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                     'summer in end-century period': 'heatindex_E85_DayMax',
                                                                     'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                                     'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},
                'number of days with daily heat index above 115 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                     'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                     'summer in end-century period': 'heatindex_E85_DayMax',
                                                                     'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                                     'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},
                'number of days with daily heat index above 125 F': {'summer in historical period': 'heatindex_HIS_DayMax',
                                                                     'summer in mid-century period': 'heatindex_M85_DayMax',
                                                                     'summer in end-century period': 'heatindex_E85_DayMax',
                                                                     'summer in change between historical and mid-century period': 'heatindex_C_M85_DMax',
                                                                     'summer in change between historical and end-century period': 'heatindex_C_E85_DMax'},

                'heating degree': {'historical period': 'hist',
                                   'mid-century period': 'rcp85_midc',
                                   'changes between historical and mid-century period': 'mid85_hist'},

                'annual total precipitation': {'historical period': 'hist',
                                               'mid-century period (RCP4.5)': 'rcp45_midc',
                                               'end-century period (RCP4.5)': 'rcp45_endc',
                                               'mid-century period (RCP8.5)': 'rcp85_midc',
                                               'end-century period (RCP8.5)': 'rcp85_endc',
                                               'change between historical and mid-century period (RCP4.5)': 'mid45_hist',
                                               'changes between historical and end-century period (RCP4.5)': 'end45_hist',
                                               'changes between historical and mid-century period (RCP8.5)': 'mid85_hist',
                                               'changes between historical and end-century period (RCP8.5)': 'end85_hist',
                                               'difference in mid-century periods (RCP4.5 and RCP8.5)': 'mid85_45',
                                               'difference in end-century periods (RCP4.5 and RCP8.5)': 'end85_45'},

                'maximum seasonal temperature': {'spring in historical period': 'hist_spring',
                                                 'spring in mid-century period (RCP8.5)': 'rcp85_midc_spring',
                                                 'spring in end-century period (RCP8.5)': 'rcp85_endc_spring',
                                                 'changes in spring between historical and mid-century period (RCP8.5)': 'mid85_hist_spring',
                                                 'changes in spring between historical and end-century period (RCP8.5)': 'end85_hist_spring',
                                                 'summer in historical period': 'hist_summer',
                                                 'summer in mid-century period (RCP8.5)': 'rcp85_midc_summer',
                                                 'summer in end-century period (RCP8.5)': 'rcp85_endc_summer',
                                                 'changes in summer between historical and mid-century period (RCP8.5)': 'mid85_hist_summer',
                                                 'changes in summer between historical and end-century period (RCP8.5)': 'end85_hist_summer',
                                                 'autumn in historical period': 'hist_autumn',
                                                 'autumn in mid-century period (RCP8.5)': 'rcp85_midc_autumn',
                                                 'autumn in end-century period (RCP8.5)': 'rcp85_endc_autumn',
                                                 'changes in autumn between historical and mid-century period (RCP8.5)': 'mid85_hist_autumn',
                                                 'changes in autumn between historical and end-century period (RCP8.5)': 'end85_hist_autumn',
                                                 'winter in historical period': 'hist_winter',
                                                 'winter in mid-century period (RCP8.5)': 'rcp85_midc_winter',
                                                 'winter in end-century period (RCP8.5)': 'rcp85_endc_winter',
                                                 'changes in winter between historical and mid-century period (RCP8.5)': 'mid85_hist_winter',
                                                 'changes in winter between historical and end-century period (RCP8.5)': 'end85_hist_winter'},

                'minimum seasonal temperature': {'spring in historical period': 'hist_spring',
                                                 'spring in mid-century period (RCP8.5)': 'rcp85_midc_spring',
                                                 'spring in end-century period (RCP8.5)': 'rcp85_endc_spring',
                                                 'changes in spring between historical and mid-century period (RCP8.5)': 'mid85_hist_spring',
                                                 'changes in spring between historical and end-century period (RCP8.5)': 'end85_hist_spring',
                                                 'summer in historical period': 'hist_summer',
                                                 'summer in mid-century period (RCP8.5)': 'rcp85_midc_summer',
                                                 'summer in end-century period (RCP8.5)': 'rcp85_endc_summer',
                                                 'changes in summer between historical and mid-century period (RCP8.5)': 'mid85_hist_summer',
                                                 'changes in summer between historical and end-century period (RCP8.5)': 'end85_hist_summer',
                                                 'autumn in historical period': 'hist_autumn',
                                                 'autumn in mid-century period (RCP8.5)': 'rcp85_midc_autumn',
                                                 'autumn in end-century period (RCP8.5)': 'rcp85_endc_autumn',
                                                 'changes in autumn between historical and mid-century period (RCP8.5)': 'mid85_hist_autumn',
                                                 'changes in autumn between historical and end-century period (RCP8.5)': 'end85_hist_autumn',
                                                 'winter in historical period': 'hist_winter',
                                                 'winter in mid-century period (RCP8.5)': 'rcp85_midc_winter',
                                                 'winter in end-century period (RCP8.5)': 'rcp85_endc_winter',
                                                 'changes in winter between historical and mid-century period (RCP8.5)': 'mid85_hist_winter',
                                                 'changes in winter between historical and end-century period (RCP8.5)': 'end85_hist_winter'},

                'wind speed': {'historical period': 'hist',
                               'mid-century period (RCP4.5)': 'rcp45_midc',
                               'end-century period (RCP4.5)': 'rcp45_endc',
                               'mid-century period (RCP8.5)': 'rcp85_midc',
                               'end-century period (RCP8.5)': 'rcp85_endc',
                               'change between historical and mid-century period (RCP4.5)': 'mid45_hist',
                               'changes between historical and end-century period (RCP4.5)': 'end45_hist',
                               'changes between historical and mid-century period (RCP8.5)': 'mid85_hist',
                               'changes between historical and end-century period (RCP8.5)': 'end85_hist',
                               'difference in mid-century periods (RCP4.5 and RCP8.5)': 'mid85_45',
                               'difference in end-century periods (RCP4.5 and RCP8.5)': 'end85_45'},
                }


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


def find_largest_rectangle(matrix):
    """
    Finds the largest rectangle inside a 2D matrix without any NaN values by iteratively removing edges with more than half NaN values.

    Args:
        matrix (pd.DataFrame): Input matrix with potential NaN values.

    Returns:
        pd.DataFrame: Submatrix containing the largest rectangle without NaN.
    """

    def edge_nan_fraction(matrix, edge):
        """Calculate the fraction of NaN values along a specified edge."""
        if edge == 'top':
            return matrix.iloc[0].isna().mean()
        elif edge == 'bottom':
            return matrix.iloc[-1].isna().mean()
        elif edge == 'left':
            return matrix.iloc[:, 0].isna().mean()
        elif edge == 'right':
            return matrix.iloc[:, -1].isna().mean()

    def remove_edge(matrix, edge):
        """Remove a specified edge from the matrix."""
        if edge == 'top':
            return matrix.iloc[1:]
        elif edge == 'bottom':
            return matrix.iloc[:-1]
        elif edge == 'left':
            return matrix.iloc[:, 1:]
        elif edge == 'right':
            return matrix.iloc[:, :-1]

    changed = True
    while changed:
        changed = False
        for edge in ['top', 'right', 'bottom', 'left']:
            if edge_nan_fraction(matrix, edge) > 0.5:
                matrix = remove_edge(matrix, edge)
                changed = True

    return matrix


def reformat_to_2d_table(data, crossmodel_indices):
    # Extract rows and columns from crossmodel_indices
    rows = [int(index[1:].split('C')[0]) for index in crossmodel_indices]
    cols = [int(index.split('C')[1]) for index in crossmodel_indices]

    # Create a DataFrame indexed by rows and columns
    df = pd.DataFrame({'Row': rows, 'Col': cols, 'Value': data})

    # Pivot the DataFrame to create a 2D table
    pivot_table = df.pivot(index='Row', columns='Col', values='Value')

    # Fill missing values with NaN and sort rows and columns
    pivot_table = pivot_table.sort_index().sort_index(axis=1)
    pivot_table = find_largest_rectangle(pivot_table)
    return pivot_table


def merge_two_figures(figure1, figure2):
    # Ensure the inputs are valid images
    if not isinstance(figure1, Image.Image) or not isinstance(figure2, Image.Image):
        raise ValueError("Both inputs must be PIL Image instances.")

    new_width = figure1.width + figure2.width
    assert figure1.height == figure2.height, "The two figures must have the same height"
    new_height = figure1.height
    merged_figure = Image.new("RGB", (new_width, new_height))

    # Paste images side by side
    merged_figure.paste(figure1, (0, 0))
    merged_figure.paste(figure2, (figure1.width, 0))
    return merged_figure


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two rectangles."""
    x1 = max(box1[0][0], box2[0][0])
    y1 = max(box1[0][1], box2[0][1])
    x2 = min(box1[2][0], box2[2][0])
    y2 = min(box1[2][1], box2[2][1])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2][0] - box1[0][0]) * (box1[2][1] - box1[0][1])
    box2_area = (box2[2][0] - box2[0][0]) * (box2[2][1] - box2[0][1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def filter_names(llm, names, curr_city):
    names = llm.query_llm(step='filter_names', content={'list': names, 'curr_city': curr_city}, assistant=False, verbose=False)
    try:
        names_filtered = ast.literal_eval(names)
    except:
        names_filtered = names
    return names_filtered


def print_qa(qa):
    question, rephrased_question, filled_values, data_var1, correct_answer, incorrect_answer, latlong1 = qa['question'], qa['rephrased_question'], qa['filled_values'], qa['data_var1'], qa['correct_answer'], qa['incorrect_answers'], qa['latlong1']
    data_var2 = qa['data_var2'] if 'data_var2' in qa else None

    print(f'{Colors.OKGREEN}Question:{Colors.ENDC}')
    print(question)
    print(f'{Colors.OKGREEN}Rephrased question:{Colors.ENDC}')
    print(rephrased_question)
    print(f'{Colors.OKGREEN}Filled values:{Colors.ENDC}')
    print(filled_values)
    print(f'{Colors.OKGREEN}Latlong1:{Colors.ENDC}')
    print(latlong1)

    print(f'{Colors.OKGREEN}Data 1:{Colors.ENDC}')
    print(data_var1)
    if data_var2 is not None:
        print(f'{Colors.OKGREEN}Data 2:{Colors.ENDC}')
        print(data_var2)

    print(f'{Colors.OKGREEN}Correct answers:{Colors.ENDC}')
    print(json.dumps(correct_answer, indent=4))
    print(f'{Colors.OKGREEN}Incorrect answers:{Colors.ENDC}')
    print(json.dumps(incorrect_answer, indent=4))



if __name__ == '__main__':
    """
    This main function helps you visualize all possible time frames of a given dataset
    """
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--dataset', type=str, default="FireWeatherIndex_Wildfire", help='Name of the dataset')
    cmd_args = parser.parse_args()

    data_path = './data/climrr/' + cmd_args.dataset + '.csv'
    print('Looking at all possible time frames of the dataset' + data_path)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(data_path)
    # Print all column names
    print("Column names:", df.columns.tolist())