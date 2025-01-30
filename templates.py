import random
import re
import argparse
import itertools
import json

import utils


visual_qa_mode = {"Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?": 'block',
                  "What is the spatial variation of {climate_variable1} in {location1} during {time_frame1}?": None,
                  "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?": 'block',
                  "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?": 'region',
                  "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?": None,
                  "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?": None,
                  "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?": 'block2',
                  "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1}?": None,
                  "How does the seasonal variation of {climate_variable1} in {location1} compare to that in {location2} for {time_frame1}?": None}


class TemplateQuestionManager:
    def __init__(self):
        # Store the template questions
        self.questions = {
            "basic": [
                "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?",
                "What is the spatial variation of {climate_variable1} in {location1} during {time_frame1}?",
                "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?",
                "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?",
                "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?",
                "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?",
                "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?",
                "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1}?",
                "How does the seasonal variation of {climate_variable1} in {location1} compare to that in {location2} for {time_frame1}?"
            ],
            "harder": [
                "What is the relationship between {population density} and {climate_variable1} risk in {location1} during {time_frame1}?",
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
        
        self.climate_variables = utils.climate_variables
        self.allowed_time_frames = utils.allowed_time_frames

        # the following dictionary contains data for changes and differences between time frames, which could be used as ground truth answers
        self.full_time_frames = utils.full_time_frames


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


def generate_all_combinations(cmd_args, template_question_manager, location_picker):
    """
    Iterate through all possible combinations of variables in each template question
    and generate formatted questions.
    """
    # Set iterator for "basic" difficulty
    iterator = template_question_manager.set_iterator(difficulty="basic")
    data_collections = []

    for i, (question, variables) in enumerate(iterator):
        if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
            break

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
        for combination in variable_combinations:
            if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
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
                    if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
                        break

                    if 'season' in question or 'seasonal' in question:
                        if time_frame.split('-')[0] not in ['spring', 'summer', 'autumn', 'winter']:
                            continue
                        elif time_frame.split('-')[0] == 'spring':  # keep only one here, because one question already actually covers all four seasons
                            time_frame = ' '.join(name.split()[2:])
                        else:
                            continue

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
                            # Ensure time causality
                            if 'mid-century' in time_frame and 'historical' in time_frame2:
                                continue
                            if 'end-century' in time_frame and ('historical' in time_frame2 or 'mid-century' in time_frame2):
                                continue

                            filled_values['time_frame2'] = time_frame2

                            # Format the question
                            formatted_question = question.format(**filled_values)
                            data_collections.append({'question': formatted_question, 'filled_values': filled_values.copy(), 'template': question})
                            if cmd_args.verbose:
                                print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                                print(formatted_question)
                    else:
                        # If there are two different climate variables, make sure that they share the same timeframe1 if there is no timeframe2
                        if 'climate_variable2' in filled_values and 'time_frame2' not in filled_values:
                            if filled_values['time_frame1'] not in template_question_manager.allowed_time_frames[filled_values['climate_variable2']]:
                                continue

                        # Format the question
                        formatted_question = question.format(**filled_values)
                        data_collections.append({'question': formatted_question, 'filled_values': filled_values.copy(), 'template': question})
                        if cmd_args.verbose:
                            print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                            print(formatted_question)
            else:
                # Format the question
                formatted_question = question.format(**filled_values)
                data_collections.append({'question': formatted_question, 'filled_values': filled_values.copy(), 'template': question})
                if cmd_args.verbose:
                    print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                    print(formatted_question)

            # Save the data collections to a JSON file
            with open('./data/all_filled_questions.json', 'w') as json_file:
                json.dump(data_collections, json_file, indent=4)


def generate_one_for_each_template(cmd_args, template_question_manager, location_picker):
    """
    Iterate through all templates and generate one question for each template
    """
    # Set iterator for "basic" difficulty
    iterator = template_question_manager.set_iterator(difficulty="basic")
    data_collections = []

    for i, (question, variables) in enumerate(iterator):
        # Get a random question and the required variables for "basic" difficulty
        print(f'{utils.Colors.OKGREEN}{"Question Template:"}{utils.Colors.ENDC}')
        print(question)
        variables.sort()
        print(f'{utils.Colors.OKGREEN}{"Required Variables:"}{utils.Colors.ENDC}')
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
                    done = False
                    while not done:
                        filled_values['time_frame1'] = random.choice(list(template_question_manager.allowed_time_frames[filled_values['climate_variable1']].keys()))
                        if 'season' in question or 'seasonal' in question:
                            if filled_values['time_frame1'].split('-')[0] not in ['spring', 'summer', 'autumn', 'winter']:
                                done = False
                                continue
                            filled_values['time_frame1'] = ' '.join(filled_values['time_frame1'].split()[2:])

                        if 'climate_variable2' in filled_values:
                            if filled_values['time_frame1'] in template_question_manager.full_time_frames[filled_values['climate_variable2']]:
                                if 'season' in question or 'seasonal' in question:
                                    if filled_values['time_frame1'].split('-')[0] not in ['spring', 'summer', 'autumn', 'winter']:
                                        done = False
                                        continue
                                done = True
                            else:
                                done = False
                        else:
                            done = True
                else:
                    assert 'time_frame1' in filled_values, "time_frame1 must be set before time_frame2"

                    # Filter time_frame2 to avoid conflicting RCP scenarios
                    time_frame1_rcp = 'RCP4.5' if 'RCP4.5' in filled_values['time_frame1'] else 'RCP8.5' if 'RCP8.5' in filled_values['time_frame1'] else None
                    done = False
                    while not done:
                        time_frame2 = random.choice([var for var in template_question_manager.allowed_time_frames[filled_values['climate_variable1']].keys()])
                        filled_values['time_frame2'] = time_frame2

                        done = True
                        if time_frame2 == filled_values['time_frame1']:
                            done = False
                        if time_frame1_rcp == 'RCP4.5' and 'RCP8.5' in time_frame2:
                            done = False
                        if time_frame1_rcp == 'RCP8.5' and 'RCP4.5' in time_frame2:
                            done = False
                        if 'mid-century' in filled_values['time_frame1'] and 'historical' in filled_values['time_frame2']:
                            done = False
                        if 'end-century' in filled_values['time_frame1'] and ('historical' in filled_values['time_frame2'] or 'mid-century' in filled_values['time_frame2']):
                            done = False


        # Format the question
        formatted_question = question.format(**filled_values)
        data_collections.append({'question': formatted_question, 'filled_values': filled_values.copy(), 'template': question})
        if cmd_args.verbose:
            print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
        print(formatted_question)

    # Save the data collections to a JSON file
    with open('./data/test_filled_questions.json', 'w') as json_file:
        json.dump(data_collections, json_file, indent=4)


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--geoscale', type=str, default="city", help='Select the geographical scale for the location: city, county, or state')
    parser.add_argument('--mode', type=str, default="random", help='Select the mode: random or iterate or test')
    parser.add_argument('--max', type=int, default=-1, help='Select the maximum number of questions to generate if in iterate mode. Default is -1, which generates all possible questions.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    template_question_manager = TemplateQuestionManager()
    location_picker = LocationPicker()
    location_picker.load_data()

    if cmd_args.mode == 'random':
        generate_one_random_question(cmd_args, template_question_manager, location_picker)
    elif cmd_args.mode == 'test':
        generate_one_for_each_template(cmd_args, template_question_manager, location_picker)
    else:
        generate_all_combinations(cmd_args, template_question_manager, location_picker)

