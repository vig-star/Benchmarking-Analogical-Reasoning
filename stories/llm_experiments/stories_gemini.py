import os
import json
import requests
from dotenv import load_dotenv
import time
import logging

load_dotenv()

# Base Gemini API configuration
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_endpoint = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
headers = {
    "Content-Type": "application/json"
}

# Check if the API key is set
if not gemini_api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

id = '0613'
no = 'orig'

# Load JSON dictionary
with open(f'all_tasks_dict_{no}.json', 'r') as f:
    story_dict = json.load(f)

gemini_responses = {}


def make_gemini_request(data, max_retries=3, retry_delay=5):
    """Makes a request to the Gemini API with retries."""
    headers["x-goog-api-key"] = gemini_api_key
    for attempt in range(max_retries):
        try:
            response = requests.post(
                gemini_endpoint, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds... Attempt {attempt + 2}/{max_retries}")
                time.sleep(retry_delay)
            else:
                logging.error(f"Max retries reached, request failed: {e}")
                return {"error": f"Max retries reached, request failed with error {e}"}

    return {"error": "Max retries reached."}


# Iterate through the stories
for k in story_dict:

    gemini_responses[k] = {}
    story_1 = story_dict[k]['Story_1']
    story_a = story_dict[k]['Story_A']
    story_b = story_dict[k]['Story_B']

    # Prompts (unchanged from original)
    prompt_1 = f"""
    Consider the following story:

    Story 1: {story_1}

    Now consider two more stories:

    Story A: {story_a}

    Story B: {story_b}

    Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?
    """

    prompt_2 = f"""
    Consider the following story:

    Story 1: {story_1}

    Now consider two more stories:

    Story A: {story_b}

    Story B: {story_a}

    Which of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?
    """

    # Request payloads for Gemini
    data_1 = {
        "contents": [{
            "parts": [{"text": prompt_1}]
        }]
    }

    data_2 = {
        "contents": [{
            "parts": [{"text": prompt_2}]
        }]
    }

    # Make the API calls
    response_1 = make_gemini_request(data_1)
    if "error" not in response_1:
        gemini_responses[k]['order_1'] = response_1['candidates'][0]['content']['parts'][0]['text']
    else:
        gemini_responses[k]['order_1'] = response_1['error']

    response_2 = make_gemini_request(data_2)
    if "error" not in response_2:
        gemini_responses[k]['order_2'] = response_2['candidates'][0]['content']['parts'][0]['text']
    else:
        gemini_responses[k]['order_2'] = response_2['error']

# Save the responses to a JSON file
json_string = json.dumps(gemini_responses, indent=2)
output_path = os.path.join('..', 'llm_results', f'gemini_{id}_responses_dict_{no}.json')
with open(output_path, 'w') as json_f:
    json_f.write(json_string)
