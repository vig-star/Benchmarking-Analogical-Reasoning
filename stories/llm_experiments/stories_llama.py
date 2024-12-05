import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

id = 'llama'
no = 'new'

data_file = f'all_tasks_dict_{no}.json'

with open(data_file, 'r') as f:
    story_dict = json.load(f)

llama_responses = {}

for k in story_dict:
    llama_responses[k] = {}
    story_1 = story_dict[k]['Story_1']
    story_a = story_dict[k]['Story_A']
    story_b = story_dict[k]['Story_B']

    prompt_1 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
    prompt_2 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"

    inputs = tokenizer(prompt_1, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response_1 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llama_responses[k]['order_1'] = response_1.strip()

    inputs = tokenizer(prompt_2, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response_2 = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llama_responses[k]['order_2'] = response_2.strip()

json_string = json.dumps(llama_responses, indent=2)
with open(f'llm_results/llama_{id}_responses_dict_{no}.json', 'w') as json_f:
    json_f.write(json_string)
