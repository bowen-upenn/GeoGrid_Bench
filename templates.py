import random
import re
import argparse
import itertools
import json

import utils


visual_qa_mode = {"Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?": 'block',
                  "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?": 'block',
                  "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?": 'region',
                  "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?": None,
                  "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?": None,
                  "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?": 'block',
                  "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1} and {time_frame2}?": None,
                  "How does the seasonal variation of {climate_variable1} in {location1} compare to that in {location2} for {time_frame1}?": None}


class TemplateQuestionManager:
    def __init__(self):
        # Store the template questions
        self.questions = {
            "basic": [
                "Which region in the {location1} experienced the largest increase in {climate_variable1} during {time_frame1}?",
                "How has {climate_variable1} changed between {time_frame1} and {time_frame2} in the {location1}?",
                "What is the correlation between {climate_variable1} and {climate_variable2} in the {location1} during {time_frame1}?",
                "What is the seasonal variation of {climate_variable1} in {location1} during {time_frame1}?",
                "Which season in {time_frame1} saw the highest levels of {climate_variable1} in {location1}?",
                "How does {climate_variable1} compare between {location1} and {location2} during {time_frame1}?",
                "Which of {location1} or {location2} experienced a greater change in {climate_variable1} throughout {time_frame1} and {time_frame2}?",
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


    def load_data(self, level, climate_related=True):
        """
        Load data from the files into memory.
        """
        if climate_related:
            assert level == 'city', "Climate-related data is only available at the city level."
            with open("./data/top_us_cities_climate_related.json", "r") as cities_file:
                self.cities = json.load(cities_file)
        else:
            if level == 'state':
                # Load states
                with open("./data/all_us_states.txt", "r") as states_file:
                    self.states = [line.strip() for line in states_file.readlines()]
            elif level == 'county':
                # Load counties
                with open("./data/all_us_counties.txt", "r") as counties_file:
                    self.counties = [line.strip() for line in counties_file.readlines()]
            elif level == 'city':
                # Load cities
                with open("./data/top_us_cities.txt", "r") as cities_file:
                    self.cities = [line.strip() for line in cities_file.readlines()]


    def get_all_locations(self, level='city', climate_variable=None):
        """
        Get all locations at the specified level.
        """
        assert level in ['state', 'county', 'city'], "Invalid level. Choose 'state', 'county', or 'city'."

        if climate_variable:
            assert level == 'city', "Climate-related data is only available at the city level."
            return self.cities[climate_variable]

        if level == 'city':
            return self.cities
        elif level == 'county':
            return self.counties
        elif level == 'state':
            return self.states


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


def generate_all_combinations(cmd_args, template_question_manager, location_picker, climate_related=True):
    """
    Iterate through all possible combinations of variables in each template question
    and generate formatted questions.
    Additionally:
      (1) The order of variable_combinations is randomly shuffled.
      (2) Duplicate orderings of the same values (for variables of the same kind) are removed.
      (3) Questions are grouped by template in a dictionary.
      (4) Each template is limited to a maximum of 1000 questions.
    """
    # Set iterator for "basic" difficulty
    iterator = template_question_manager.set_iterator(difficulty="basic")

    # Dictionary to store questions grouped by template
    template_questions = {}

    for i, (question, variables) in enumerate(iterator):
        if cmd_args.max != -1 and sum(len(q_list) for q_list in template_questions.values()) >= cmd_args.max:
            break

        # Ensure climate_variable1 is processed first
        variables.sort(key=lambda x: x != 'climate_variable1')

        variable_options = {}
        for variable in variables:
            if variable == 'climate_variable1':
                # This is a list of climate variable keys.
                variable_options['climate_variable1'] = list(template_question_manager.climate_variables.keys())
            elif variable == 'climate_variable2':
                variable_options['climate_variable2'] = list(template_question_manager.climate_variables.keys())
            elif variable == 'location1':
                if climate_related:
                    # Build a dictionary mapping each climate variable to its list of locations.
                    cv_keys = variable_options['climate_variable1']
                    variable_options['location1'] = {
                        cv: location_picker.get_all_locations(level=cmd_args.geoscale, climate_variable=cv)
                        for cv in cv_keys
                    }
                else:
                    # Non climate-related: get a simple list.
                    variable_options['location1'] = location_picker.get_all_locations(level=cmd_args.geoscale)
            elif variable == 'location2':
                if climate_related:
                    # For climate-related, share the same dictionary as location1.
                    variable_options['location2'] = variable_options['location1']
                else:
                    variable_options['location2'] = variable_options['location1']
            elif variable == 'time_frame1':
                pass  # Handled later
            elif variable == 'time_frame2':
                pass  # Handled later
            else:
                raise ValueError(f"Variable not implemented: {variable}")

        # Prepare a list of variables to combine for non-time and non-location variables.
        non_time_vars = [var for var in variables if var not in ['time_frame1', 'time_frame2', 'location1', 'location2']]

        # Generate all combinations for the non-location, non-time variables.
        variable_combinations = list(itertools.product(
            *[variable_options[var] for var in non_time_vars]
        ))

        # Now, for each combination (which includes a chosen climate_variable1),
        # we iterate over the locations corresponding to that climate variable.
        final_combinations = []
        for comb in variable_combinations:
            filled_values = dict(zip(non_time_vars, comb))
            # If climate_related and the template requires a location,
            # then use the climate_variable1 value to lookup the available locations.
            if climate_related and 'climate_variable1' in filled_values:
                cv_value = filled_values['climate_variable1']
                available_locations = variable_options['location1'].get(cv_value, [])
                # If the template expects two locations (location1 and location2):
                if 'location2' in variables:
                    # Iterate over all distinct pairs of locations from the available list.
                    for loc1, loc2 in itertools.combinations(available_locations, 2):
                        temp_filled = filled_values.copy()
                        temp_filled['location1'] = loc1
                        temp_filled['location2'] = loc2
                        final_combinations.append(temp_filled)
                else:
                    # Otherwise, iterate over all available locations.
                    for loc in available_locations:
                        temp_filled = filled_values.copy()
                        temp_filled['location1'] = loc
                        final_combinations.append(temp_filled)
            else:
                # If not climate-related, locations were already provided as lists.
                # Add combinations for location variables directly.
                final_combinations.append(filled_values)

        variable_combinations = final_combinations

        # Randomly shuffle the order of the combinations
        random.shuffle(variable_combinations)

        # Track unique variable combinations (avoiding order-dependent duplicates)
        seen_combinations = set()
        template_questions[question] = []

        for combination in variable_combinations:
            if cmd_args.max != -1 and sum(len(q_list) for q_list in template_questions.values()) >= cmd_args.max:
                break
            if len(template_questions[question]) >= 200:  # Limit to 200 per template
                break

            # Use the combination directly since it is already a dictionary of filled values.
            filled_values = combination.copy()

            # Sort values of interchangeable variables (e.g. for climate and location) to avoid duplicate orderings.
            for var1, var2 in [('climate_variable1', 'climate_variable2'), ('location1', 'location2')]:
                if var1 in filled_values and var2 in filled_values:
                    sorted_values = tuple(sorted([filled_values[var1], filled_values[var2]]))
                    filled_values[var1], filled_values[var2] = sorted_values

            # Check for duplicate variable combinations using a sorted tuple of the items.
            frozen_combination = tuple(sorted(filled_values.items()))
            if frozen_combination in seen_combinations:
                continue  # Skip duplicate orderings
            seen_combinations.add(frozen_combination)

            # Validate dependent variable constraints
            if 'climate_variable2' in filled_values and filled_values['climate_variable2'] == filled_values['climate_variable1']:
                continue  # Skip invalid combinations
            if 'location2' in filled_values and filled_values['location2'] == filled_values['location1']:
                continue  # Skip invalid combinations

            # Process time frames if needed.
            if 'time_frame1' in variables:
                climate_var = filled_values.get('climate_variable1')
                if 'season' in question or 'seasonal' in question:
                    allowed_time_frames = {}
                    for cv in template_question_manager.climate_variables.keys():
                        for tf in template_question_manager.allowed_time_frames[cv]:
                            if tf.split(' ')[0] == 'spring':  # seasonal questions cover all four seasons
                                allowed_time_frames[tf] = template_question_manager.allowed_time_frames[cv][tf]
                        if len(allowed_time_frames) > 0:
                            filled_values['climate_variable1'] = cv
                            break
                else:
                    allowed_time_frames = template_question_manager.allowed_time_frames[climate_var]

                # Randomly shuffle the allowed time frames.
                allowed_time_frames = list(allowed_time_frames.items())
                random.shuffle(allowed_time_frames)
                allowed_time_frames = dict(allowed_time_frames)

                # Loop over allowed time frames for time_frame1.
                for time_frame in allowed_time_frames:
                    if cmd_args.max != -1 and sum(len(q_list) for q_list in template_questions.values()) >= cmd_args.max:
                        break
                    if len(template_questions[question]) >= 1000:
                        break

                    if 'season' in question or 'seasonal' in question:
                        filled_values['time_frame1'] = ' '.join(time_frame.split(' ')[2:])
                    else:
                        filled_values['time_frame1'] = time_frame
                    time_frame1_rcp = (
                        'RCP4.5' if 'RCP4.5' in time_frame
                        else 'RCP8.5' if 'RCP8.5' in time_frame
                        else None
                    )

                    # If a second time frame is required, process it.
                    if 'time_frame2' in variables:
                        for time_frame2 in allowed_time_frames:
                            if time_frame2 == filled_values['time_frame1']:
                                continue
                            # Avoid conflicting RCP scenarios.
                            if time_frame1_rcp == 'RCP4.5' and 'RCP8.5' in time_frame2:
                                continue
                            if time_frame1_rcp == 'RCP8.5' and 'RCP4.5' in time_frame2:
                                continue

                            filled_values['time_frame2'] = time_frame2
                            sorted_time_frames = tuple(sorted([filled_values['time_frame1'], filled_values['time_frame2']]))
                            filled_values['time_frame1'], filled_values['time_frame2'] = sorted_time_frames

                            frozen_combination = tuple(sorted(filled_values.items()))
                            if frozen_combination in seen_combinations:
                                continue
                            seen_combinations.add(frozen_combination)

                            formatted_question = question.format(**filled_values)
                            template_questions[question].append({
                                'question': formatted_question,
                                'filled_values': filled_values.copy()
                            })
                    else:
                        formatted_question = question.format(**filled_values)
                        template_questions[question].append({
                            'question': formatted_question,
                            'filled_values': filled_values.copy()
                        })
            else:
                formatted_question = question.format(**filled_values)
                template_questions[question].append({
                    'question': formatted_question,
                    'filled_values': filled_values.copy()
                })

        # Ensure no more than 1000 questions per template
        template_questions[question] = template_questions[question][:1000]

    # Save the data collections to a JSON file
    with open('./data/all_filled_questions.json', 'w') as json_file:
        json.dump(template_questions, json_file, indent=4)

    for template, questions in template_questions.items():
        print(f"Number of questions for template '{template}': {len(questions)}")
    print('Questions generated and saved to all_filled_questions.json')

    return template_questions


def generate_one_for_each_template(cmd_args, template_question_manager, location_picker):
    """
    Iterate through all possible combinations of variables in each template question
    and generate formatted questions. Now stops at the first valid example per template.
    For seasonal questions, invalid time frames are filtered out before looping.
    """
    # Set iterator for "basic" difficulty
    iterator = template_question_manager.set_iterator(difficulty="basic")
    data_collections = []

    for i, (question, variables) in enumerate(iterator):
        if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
            break

        # Use a flag to track if we already found a valid example for this template
        found_example = False

        # Generate options for each variable as before
        variable_options = {}
        for variable in variables:
            if variable == 'climate_variable1':
                variable_options['climate_variable1'] = list(template_question_manager.climate_variables.keys())
            elif variable == 'climate_variable2':
                variable_options['climate_variable2'] = list(template_question_manager.climate_variables.keys())
            elif variable == 'location1':
                variable_options['location1'] = location_picker.get_all_locations(level=cmd_args.geoscale)
            elif variable == 'location2':
                variable_options['location2'] = variable_options['location1']
            elif variable == 'time_frame1':
                # We will use allowed_time_frames later based on the chosen climate_variable1.
                pass
            elif variable == 'time_frame2':
                # time_frame2 uses the same valid time frames as time_frame1
                pass
            else:
                raise ValueError(f"Variable not implemented: {variable}")

        # Generate all combinations of variable values (except time frames, which we handle later)
        variable_combinations = itertools.product(
            *[variable_options[var] for var in variables if var not in ['time_frame1', 'time_frame2']]
        )

        for combination in variable_combinations:
            if found_example:
                break

            if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
                break

            # Reconstruct filled_values with the non-time frame variables
            filled_values = dict(zip([var for var in variables if var not in ['time_frame1', 'time_frame2']], combination))

            # Validate dependent variable constraints
            if 'climate_variable2' in filled_values:
                if filled_values['climate_variable2'] == filled_values['climate_variable1']:
                    continue  # Skip invalid combinations

            if 'location2' in filled_values:
                if filled_values['location2'] == filled_values['location1']:
                    continue  # Skip invalid combinations

            # Now process time frames if needed
            if 'time_frame1' in variables:
                # --- Filter allowed time frames up front for seasonal questions ---
                climate_var = filled_values['climate_variable1']
                if 'season' in question or 'seasonal' in question:
                    allowed_time_frames = []
                    for climate_var in template_question_manager.climate_variables.keys():
                        for tf in template_question_manager.allowed_time_frames[climate_var]:
                            if tf.split(' ')[0] == 'spring':    # seasonal questions will cover all four seasons so no duplicates needed here
                                allowed_time_frames.append(' '.join(tf.split(' ')[2:]))  # Remove the first word and store the rest
                        if len(allowed_time_frames) > 0:
                            filled_values['climate_variable1'] = climate_var
                            break
                else:
                    allowed_time_frames = template_question_manager.allowed_time_frames[climate_var]

                # Iterate over the (possibly filtered) allowed time frames for time_frame1.
                for time_frame in allowed_time_frames:
                    if found_example:
                        break
                    if cmd_args.max != -1 and len(data_collections) >= cmd_args.max:
                        break

                    filled_values['time_frame1'] = time_frame
                    time_frame1_rcp = (
                        'RCP4.5' if 'RCP4.5' in time_frame
                        else 'RCP8.5' if 'RCP8.5' in time_frame
                        else None
                    )

                    # If a second time frame is required, process it here.
                    if 'time_frame2' in variables:
                        for time_frame2 in template_question_manager.allowed_time_frames[climate_var]:
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

                            formatted_question = question.format(**filled_values)
                            data_collections.append({
                                'question': formatted_question,
                                'filled_values': filled_values.copy(),
                                'template': question
                            })
                            if cmd_args.verbose:
                                print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                                print(formatted_question)
                            found_example = True
                            break  # Exit the time_frame2 loop after the first valid example

                    else:
                        # For questions with a single time frame, format and store the question.
                        formatted_question = question.format(**filled_values)
                        data_collections.append({
                            'question': formatted_question,
                            'filled_values': filled_values.copy(),
                            'template': question
                        })
                        if cmd_args.verbose:
                            print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                            print(formatted_question)
                        found_example = True
                        break  # Exit the allowed_time_frames loop for time_frame1

            else:
                # If there is no time_frame1 variable, simply format and store the question.
                formatted_question = question.format(**filled_values)
                data_collections.append({
                    'question': formatted_question,
                    'filled_values': filled_values.copy(),
                    'template': question
                })
                if cmd_args.verbose:
                    print(f'{utils.Colors.OKGREEN}Filled Question:{utils.Colors.ENDC}')
                    print(formatted_question)
                found_example = True
                break  # Exit the variable_combinations loop

        # Save the data collections to a JSON file (can be done after processing each template)
        with open('./data/test_filled_questions.json', 'w') as json_file:
            json.dump(data_collections, json_file, indent=4)


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command line arguments')
    parser.add_argument('--geoscale', type=str, default="city", help='Select the geographical scale for the location: city, county, or state')
    parser.add_argument('--mode', type=str, default="test", help='Select the mode: iterate or test')
    parser.add_argument('--max', type=int, default=-1, help='Select the maximum number of questions to generate if in iterate mode. Default is -1, which generates all possible questions.')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')
    cmd_args = parser.parse_args()

    template_question_manager = TemplateQuestionManager()
    location_picker = LocationPicker()
    location_picker.load_data(cmd_args.geoscale)

    if cmd_args.mode == 'random':
        generate_one_random_question(cmd_args, template_question_manager, location_picker)
    elif cmd_args.mode == 'test':
        generate_one_for_each_template(cmd_args, template_question_manager, location_picker)
    else:
        generate_all_combinations(cmd_args, template_question_manager, location_picker)

