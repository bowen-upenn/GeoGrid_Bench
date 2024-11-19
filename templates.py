import random
import re
import utils


class TemplateQuestionManager:
    def __init__(self):
        # Store the template questions
        self.questions = {
            "basic": [
                "Which region in the {location} experienced the largest increase in {climate_variable} from {time_frame}?",
                "What is the correlation between {climate_variable1} and {climate_variable2} in the {location} during {time_frame}?",
                "What is the relationship between population density and {climate_variable} risk in {location} by {time_frame}?",
                "How has {climate_variable} changed between {time_frame1} and {time_frame2} in the {location}?",
                "What is the seasonal variation of {climate_variable} in {location} during {time_frame}?",
                "Which season in {time_frame} saw the highest levels of {climate_variable} in {location}?",
                "How does {climate_variable} compare between {location1} and {location2} during {time_frame}?",
                "Which of {location1} or {location2} experienced a greater change in {climate_variable} throughout {time_frame}?",
                "How does the seasonal variation of {climate_variable} in {location1} compare to that in {location2} within {location} for {time_frame}?"
            ],
            "harder": [
                "How does the seasonal variation of {climate_variable1} in {location1} compare to {climate_variable2} in {location2} for {time_frame}?",
                "What are the differences in the annual trends of {climate_variable} between {location1} and {location2} in relation to the {socioeconomic_variable} for {time_frame}?",
                "How do trends in {elderly_population_rate} in {location1} and {location2} relate to the seasonal variation in {climate_variable} for {time_frame}?",
                "How do the rates of {no_vehicle_rate} and {single_unit_housing_rate} differ between {location1} and {location2} in relation to {climate_variable} patterns over {time_frame}?",
                "To what extent does the {elderly_population_rate} in {location1} align with trends in {internet_access_rate} in {location2} and how do these socio-economic variables interact with changes in {climate_variable}?"
            ]
        }

        self.all_variables = {
            "common": ['specified_area', 'climate_variable', 'climate_variable1', 'climate_variable2', 'time_frame', 'location', 'location1', 'location2'],
            "harder": ['socioeconomic_variable', 'elderly_population_rate', 'no_vehicle_rate', 'single_unit_housing_rate', 'internet_access_rate']
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
            yield template


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


    def choose_random_location(self, level=''):
        """
        Randomly chooses either a state, county, or city name based on the specified parameters.
        For counties, formats the name appropriately if needed and appends ', USA'.
        """
        assert level in ['state', 'county', 'city'], "Invalid level. Choose 'state', 'county', or 'city'."

        if level == 'city':
            # Pick a random city
            chosen_city = random.choice(self.cities)
            return f"{chosen_city}, USA"

        elif level == 'county':
            # Pick a random county
            chosen_county = random.choice(self.counties)

            # Handle special format (colon separating multiple states)
            if ":" in chosen_county:
                county_name, states_after_colon = map(str.strip, chosen_county.split(":"))
                # Randomly pick one of the states listed after the colon
                state = random.choice(states_after_colon.split(";")).strip()
                chosen_county = f"{county_name}, {state}"

            # Append ", USA"
            return f"{chosen_county}, USA"

        elif level == 'state':
            # Pick a random state
            chosen_state = random.choice(self.states)
            return f"{chosen_state}, USA"


# Example Usage
if __name__ == "__main__":
    template_question_manager = TemplateQuestionManager()
    location_picker = LocationPicker()
    location_picker.load_data()

    # # Random location selection
    # random_city = location_picker.choose_random_location(level='city')
    # random_county = location_picker.choose_random_location(level='county')
    # random_state = location_picker.choose_random_location(level='state')

    # Get a random question and the required variables for "basic" difficulty
    question, variables = template_question_manager.get_random_question_and_variables(difficulty="basic")
    print(f'{utils.Colors.OKGREEN}{"Random Question Template:"}{utils.Colors.ENDC}')
    print(question)
    print(f'{utils.Colors.OKGREEN}{"Required Variables:"}:{utils.Colors.ENDC}')
    print(variables)


