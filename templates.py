import random
import re
import argparse
import itertools

import utils


class TemplateQuestionManager:
    def __init__(self):
        # Store the template questions
        self.questions = {
            "basic": [
                "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?",
                "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?",
                "What is the relationship between population density and {climate_variable1} risk in {location1} during {time_frame1}?",
                "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?",
                "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?",
                "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?",
                "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?",
                "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1}?",
                "How does the seasonal variation of {climate_variable1} in {location1} compare to that in {location2} within {location1} for {time_frame1}?"
            ],
            "harder": [
                "How does the seasonal variation of {climate_variable1} in {location1} compare to {climate_variable2} in {location2} for {time_frame1}?",
                "What are the differences in the annual trends of {climate_variable1} between {location1} and {location2} in relation to the {socioeconomic_variable} for {time_frame1}?",
                "How do trends in {elderly_population_rate} in {location1} and {location2} relate to the seasonal variation in {climate_variable1} for {time_frame1}?",
                "How do the rates of {no_vehicle_rate} and {single_unit_housing_rate} differ between {location1} and {location2} in relation to {climate_variable1} patterns over {time_frame1}?",
                "To what extent does the {elderly_population_rate} in {location1} align with trends in {internet_access_rate} in {location2} and how do these socio-economic variables interact with changes in {climate_variable}?"
            ]
        }

        self.all_variables = {
            "common": ['climate_variable1', 'climate_variable2', 'time_frame1', 'time_frame2', 'location1', 'location2'],
            "harder": ['socioeconomic_variable', 'elderly_population_rate', 'no_vehicle_rate', 'single_unit_housing_rate', 'internet_access_rate']
        }
        
        self.climate_variables = {'maximum annual temperature': './data/climrr/AnnualTemperatureMaximum.csv',
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

        self.allowed_time_frames = {'maximum annual temperature': {'historical period': 'hist',
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
        self.full_time_frames = {'maximum annual temperature': {'historical period': 'hist',
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


    def set_iterator(self, difficulty="basic"):
        """Validate the difficulty level."""
        if difficulty not in self.questions:
            raise ValueError("Invalid difficulty level. Choose 'basic' or 'harder'.")
        return self._question_generator(self.questions[difficulty])


    def _question_generator(self, questions):
        """
        Yield questions one at a time, formatted with placeholders if applicable.
        """
        for template in questions:
            required_variables = re.findall(r"\{(.*?)\}", template)
            yield template, required_variables


    def get_random_question_and_variables(self, difficulty="basic"):
        """
        Randomly select a question template and return the template along with required variables.
        Args:
            difficulty (str): "basic" or "harder", default is "basic".
        Returns:
            tuple: (question_template, required_variables)
        """
        if difficulty not in self.questions:
            raise ValueError("Invalid difficulty level. Choose 'basic' or 'harder'.")

        # Randomly select a template
        template = random.choice(self.questions[difficulty])

        # Extract all placeholders from the template using regex
        required_variables = re.findall(r"\{(.*?)\}", template)

        return template, required_variables


class LocationPicker:
    def __init__(self):
        # Storage for loaded data
        self.states = []
        self.counties = []
        self.cities = []

    def load_data(self):
        """
        Load data from the files into memory.
        """
        try:
            # Load states
            with open("./data/all_us_states.txt", "r") as states_file:
                self.states = [line.strip() for line in states_file.readlines()]

            # Load counties
            with open("./data/all_us_counties.txt", "r") as counties_file:
                self.counties = [line.strip() for line in counties_file.readlines()]

            # Load cities
            with open("./data/top_us_cities.txt", "r") as cities_file:
                self.cities = [line.strip() for line in cities_file.readlines()]

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading data: {e}. Ensure all required files are present.")


    def get_all_locations(self, level='city'):
        """
        Get all locations at the specified level.
        """
        assert level in ['state', 'county', 'city'], "Invalid level. Choose 'state', 'county', or 'city'."

        if level == 'city':
            return self.cities
        elif level == 'county':
            return self.counties
        elif level == 'state':
            return


    def choose_random_location(self, level='city', exclude=None):
        """
        Randomly chooses either a state, county, or city name based on the specified parameters.
        For counties, formats the name appropriately if needed and appends ', USA'.
        """
        assert level in ['state', 'county', 'city'], "Invalid level. Choose 'state', 'county', or 'city'."

        if level == 'city':
            # Pick a random city
            if exclude is None:
                chosen_city = random.choice(self.cities)
            else:
                chosen_city = random.choice([city for city in self.cities if city != exclude])
            return chosen_city

        elif level == 'county':
            # Pick a random county
            if exclude is None:
                chosen_county = random.choice(self.counties)
            else:
                chosen_county = random.choice([county for county in self.counties if county != exclude])

            # Handle special format (colon separating multiple states)
            if ":" in chosen_county:
                county_name, states_after_colon = map(str.strip, chosen_county.split(":"))
                # Randomly pick one of the states listed after the colon
                state = random.choice(states_after_colon.split(";")).strip()
                chosen_county = f"{county_name}, {state}"
            return chosen_county

        elif level == 'state':
            # Pick a random state
            if exclude is None:
                chosen_state = random.choice(self.states)
            else:
                chosen_state = random.choice([state for state in self.states if state != exclude])
            return chosen_state


def generate_one_random_question(cmd_args, template_question_manager, location_picker):
    """
    Find a random value for each variable in each template question
    and generate formatted questions.
    """
    # Get a random question and the required variables for "basic" difficulty
    question, variables = template_question_manager.get_random_question_and_variables(difficulty="basic")
    print(f'{utils.Colors.OKGREEN}{"Random Question Template:"}{utils.Colors.ENDC}')
    print(question)
    print(f'{utils.Colors.OKGREEN}{"Required Variables:"}:{utils.Colors.ENDC}')
    print(variables)

    # Randomly select values for the required variables
    filled_values = {}
    for variable in variables:
        if variable in ['climate_variable1', 'climate_variable2']:
            if variable == 'climate_variable1':
                filled_values['climate_variable1'] = random.choice(list(template_question_manager.climate_variables.keys()))
            else:
                assert 'climate_variable1' in filled_values, "climate_variable1 must be set before climate_variable2"
                filled_values['climate_variable2'] = random.choice(
                    [var for var in template_question_manager.climate_variables.keys() if var != filled_values['climate_variable1']]
                )

        elif variable in ['location1', 'location2']:
            if variable == 'location1':
                filled_values['location1'] = location_picker.choose_random_location(level=cmd_args.geoscale)
            else:
                assert 'location1' in filled_values, "location1 must be set before location2"
                filled_values['location2'] = location_picker.choose_random_location(level=cmd_args.geoscale, exclude=filled_values['location1'])

        elif variable in ['time_frame1', 'time_frame2']:
            if variable == 'time_frame1':
                filled_values['time_frame1'] = random.choice(list(template_question_manager.allowed_time_frames[filled_values['climate_variable1']].keys()))
            else:
                assert 'time_frame1' in filled_values, "time_frame1 must be set before time_frame2"

                # Filter time_frame2 to avoid conflicting RCP scenarios
                time_frame1_rcp = 'RCP4.5' if 'RCP4.5' in filled_values['time_frame1'] else 'RCP8.5' if 'RCP8.5' in filled_values['time_frame1'] else None
                filled_values['time_frame2'] = random.choice([
                    var for var in template_question_manager.allowed_time_frames[filled_values['climate_variable1']].keys()
                    if var != filled_values['time_frame1'] and (
                            (time_frame1_rcp == 'RCP4.5' and 'RCP8.5' not in var) or
                            (time_frame1_rcp == 'RCP8.5' and 'RCP4.5' not in var)
                    )
                ])
        else:
            raise ValueError(f"Variable not implemented: {variable}")

    # Format the question with filled values
    formatted_question = question.format(**filled_values)
    print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
    print(formatted_question)


def generate_all_combinations(cmd_args, template_question_manager, location_picker, verbose=False):
    """
    Iterate through all possible combinations of variables in each template question
    and generate formatted questions.
    """
    # Set iterator for "basic" difficulty
    iterator = template_question_manager.set_iterator(difficulty="basic")

    for i, (question, variables) in enumerate(iterator):
        if verbose:
            print(f'{utils.Colors.OKGREEN}{"Question Template:"}{utils.Colors.ENDC}')
            print(question)
            print(f'{utils.Colors.OKGREEN}{"Required Variables:"}:{utils.Colors.ENDC}')
            print(variables)

        # Generate all possible values for each variable
        variable_options = {}
        for variable in variables:
            if variable == 'climate_variable1':
                variable_options['climate_variable1'] = list(template_question_manager.climate_variables.keys())
            elif variable == 'climate_variable2':
                variable_options['climate_variable2'] = [
                    var for var in template_question_manager.climate_variables.keys()
                ]
            elif variable == 'location1':
                variable_options['location1'] = location_picker.get_all_locations(level=cmd_args.geoscale)
            elif variable == 'location2':
                variable_options['location2'] = variable_options['location1']
            elif variable == 'time_frame1':
                # Time frames depend on the selected climate_variable1
                variable_options['time_frame1'] = [
                    list(template_question_manager.allowed_time_frames[climate_var].keys())
                    for climate_var in variable_options.get('climate_variable1', [])
                ]
            elif variable == 'time_frame2':
                # time_frame2 uses the same valid time frames as time_frame1
                variable_options['time_frame2'] = variable_options.get('time_frame1', [])
            else:
                raise ValueError(f"Variable not implemented: {variable}")

        # Generate all combinations of variable values
        variable_combinations = itertools.product(
            *[variable_options[var] for var in variables]
        )

        # Iterate through each combination and format the question
        j = 0
        for combination in variable_combinations:
            if cmd_args.max != -1 and j >= cmd_args.max:
                break

            filled_values = dict(zip(variables, combination))

            # Validate any dependent variable constraints
            if 'climate_variable2' in filled_values:
                if filled_values['climate_variable2'] == filled_values['climate_variable1']:
                    continue  # Skip invalid combinations

            if 'location2' in filled_values:
                if filled_values['location2'] == filled_values['location1']:
                    continue  # Skip invalid combinations

            # Decompose the time frames into individual questions
            if 'time_frame1' in filled_values:
                for time_frame in template_question_manager.allowed_time_frames[filled_values['climate_variable1']]:
                    filled_values['time_frame1'] = time_frame
                    time_frame1_rcp = 'RCP4.5' if 'RCP4.5' in time_frame else 'RCP8.5' if 'RCP8.5' in time_frame else None

                    if 'time_frame2' in filled_values:
                        for time_frame2 in template_question_manager.allowed_time_frames[filled_values['climate_variable1']]:
                            if time_frame2 == filled_values['time_frame1']:
                                continue
                            # Avoid conflicting RCP scenarios between time_frame1 and time_frame2
                            if time_frame1_rcp == 'RCP4.5' and 'RCP8.5' in time_frame2:
                                continue
                            if time_frame1_rcp == 'RCP8.5' and 'RCP4.5' in time_frame2:
                                continue

                            filled_values['time_frame2'] = time_frame2

                            # Format the question
                            j += 1
                            formatted_question = question.format(**filled_values)
                            print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                            print(formatted_question)
                    else:
                        # Format the question
                        j += 1
                        formatted_question = question.format(**filled_values)
                        print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                        print(formatted_question)
            else:
                # Format the question
                j += 1
                formatted_question = question.format(**filled_values)
                print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                print(formatted_question)


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--geoscale', type=str, default="city", help='Select the geographical scale for the location: city, county, or state')
    parser.add_argument('--mode', type=str, default="random", help='Select the mode: random or iterate')
    parser.add_argument('--max', type=int, default=-1, help='Select the maximum number of questions to generate if in iterate mode. Default is -1, which generates all possible questions.')
    cmd_args = parser.parse_args()

    template_question_manager = TemplateQuestionManager()
    location_picker = LocationPicker()
    location_picker.load_data()

    if cmd_args.mode == 'random':
        generate_one_random_question(cmd_args, template_question_manager, location_picker)
    else:
        generate_all_combinations(cmd_args, template_question_manager, location_picker)

