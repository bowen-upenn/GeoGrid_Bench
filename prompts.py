def prompt_to_extract_location(location_description):
    message = "Output the latitude and longitude of " + location_description + ". Output a JSON with keys 'latitude' and 'longitude'. " \
              "Use numbers only. Do not use 'E', 'W'. Use this template: ```json```"
    return message


def prompt_to_rephrase_question(question):
    message = "Rewrite this question to provide real-world context by framing it as if a specific professional is asking it in a particular situation. " \
              "Please also rephrase the question without altering the meanings, and format it as a single-string question with no additional explanation." \
              "Avoid using the word 'climate change'.\n\nHere is current version of the question: " + question
    return message


def prompt_to_filter_names(data):
    message = "Based on your common sense, which of the following are complete and correct names on maps, but avoid " + data['curr_city'] + " or its alias. " \
              "If you believe the name only misses few letters, please complete its full name. It is a location on the map of " + data['curr_city'] + "." \
              "Here is the list:\n\n" + str(data['list']) + ".\n\n" \
              "Output a Python list of strings. Do NOT use any codes to analyze the inputs. Do NOT output any other words."
    return message