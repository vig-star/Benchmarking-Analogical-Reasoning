import os
import json
import requests
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Base Claude API configuration
claude_api_key = os.getenv("ANTHROPIC_API_KEY")  # Claude API key
claude_endpoint = "https://api.anthropic.com/v1/complete"  # Anthropic API endpoint
headers = {
    "x-api-key": claude_api_key,
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"  # This is the required header
}

# Check if the API key is set
if not claude_api_key:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

id = '0613'
no = 'new'

# Load JSON dictionary
with open(f'all_tasks_dict_{no}.json', 'r') as f:
    story_dict = json.load(f)

claude_responses = {}

# Iterate through the stories
for k in story_dict:
    claude_responses[k] = {}
    story_1 = story_dict[k]['Story_1']
    story_a = story_dict[k]['Story_A']
    story_b = story_dict[k]['Story_B']

    # Prompts
    prompt_1 = f"\n\nHuman: Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?\n\nAssistant:"
    prompt_2 = f"\n\nHuman: Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?\n\nAssistant:"

    # Request payload for Claude
    data_1 = {
        "model": "claude-2",  # Use Claude-2 or Claude-Instant
        "prompt": prompt_1,
        "max_tokens_to_sample": 300,
        "temperature": 0.7
    }

    data_2 = {
        "model": "claude-2",
        "prompt": prompt_2,
        "max_tokens_to_sample": 300,
        "temperature": 0.7
    }

    # Make the API calls
    response_1 = requests.post(claude_endpoint, headers=headers, json=data_1)
    if response_1.status_code == 200:
        claude_responses[k]['order_1'] = response_1.json().get("completion")
    else:
        claude_responses[k]['order_1'] = f"Error: {response_1.status_code}, {response_1.text}"

    response_2 = requests.post(claude_endpoint, headers=headers, json=data_2)
    if response_2.status_code == 200:
        claude_responses[k]['order_2'] = response_2.json().get("completion")
    else:
        claude_responses[k]['order_2'] = f"Error: {response_2.status_code}, {response_2.text}"

json_string = json.dumps(claude_responses, indent=2)
with open(f'llm_results/claude_{id}_responses_dict_{no}.json', 'w') as json_f:
    json_f.write(json_string)
