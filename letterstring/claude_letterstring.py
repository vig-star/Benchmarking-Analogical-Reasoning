import requests
import numpy as np
import builtins
import argparse
import os
import time
import sys


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--sentence', action='store_true', help="Present problem in sentence format.")
parser.add_argument('--noprompt', action='store_true', help="Present problem without prompt.")
parser.add_argument('--newprompt', action='store_true', help="Present problem with new prompt.")
parser.add_argument('--promptstyle', help='Give a prompt style: human, minimal, hw, webb, webbplus')
parser.add_argument('--num_permuted', help="give a number of letters in the alphabet to permute from 2 to 26")
parser.add_argument('--gen', help='give gen for generalized problems or nogen for non generalized')

args = parser.parse_args()
print(args.promptstyle)

if args.promptstyle == "webb" and int(args.num_permuted) > 1:
    print("promptstyle webb can only be used with an unpermuted alphabet")
    sys.exit()

# Anthropic Claude API settings
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_endpoint = "https://api.anthropic.com/v1/complete"
headers = {
    "x-api-key": anthropic_api_key,
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"  # Add the required header
}

# Load all problems
if args.gen == 'gen':
    all_prob = np.load(f'./problems/{args.gen}/all_prob_{args.num_permuted}_7_gpt_human_alphs.npz', allow_pickle=True)['all_prob']
elif args.gen == 'nogen':
    all_prob = np.load(f'./problems/{args.gen}/all_prob_{args.num_permuted}_7_human.npz', allow_pickle=True)['all_prob']

response_dict = {}

for alph in all_prob.item().keys():
    print(alph)
    shuffled_letters = builtins.list(all_prob.item()[alph].get('shuffled_letters', []))
    shuffled_alphabet = builtins.list(all_prob.item()[alph]['shuffled_alphabet'])

    prob_types = builtins.list(all_prob.item()[alph].keys())[2:]  # Skip shuffled letters and shuffled alphabet
    N_prob_types = len(prob_types)

    alph_string = ' '.join(shuffled_alphabet)
    print(alph_string)

    # Evaluate
    N_trials_per_prob_type = 10
    all_prob_type_responses = []
    for p in range(N_prob_types):
        print(f'problem type {p + 1} of {N_prob_types}...')
        prob_type_responses = []
        for t in range(N_trials_per_prob_type):
            print(f'trial {t + 1} of {N_trials_per_prob_type}...')
            prob = all_prob.item()[alph][prob_types[p]]['prob'][t]
            prompt = ''

            if not args.noprompt:
                if args.promptstyle not in ["minimal", "hw", "webb", "webbplus"]:
                    prompt += 'Use the following alphabet to guess the missing piece.\n\n' \
                              + alph_string \
                              + '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
                elif args.promptstyle == 'minimal':
                    prompt += 'Use the following alphabet to complete the pattern.\n\n' \
                              + alph_string \
                              + '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
                elif args.promptstyle == 'hw':
                    prompt += 'Use this fictional alphabet: \n\n' \
                              + alph_string \
                              + "\n\nLet's try to complete the pattern:\n\n"
                elif args.promptstyle == "webb":
                    prompt += "Let's try to complete the pattern:\n\n"
                elif args.promptstyle == "webbplus":
                    prompt += "Let's try to complete the pattern. Just give the letters that complete the pattern and nothing else at all. Do not describe the pattern.\n\n"
            if args.sentence:
                prompt += 'If '
                for i in range(len(prob[0][0])):
                    prompt += str(prob[0][0][i])
                    if i < len(prob[0][0]) - 1:
                        prompt += ' '
                prompt += ' changes to '
                for i in range(len(prob[0][1])):
                    prompt += str(prob[0][1][i])
                    if i < len(prob[0][1]) - 1:
                        prompt += ' '
                prompt += ', then '
                for i in range(len(prob[1][0])):
                    prompt += str(prob[1][0][i])
                    if i < len(prob[1][0]) - 1:
                        prompt += ' '
                prompt += ' should change to '
            else:
                prompt += '['
                for i in range(len(prob[0][0])):
                    prompt += str(prob[0][0][i])
                    if i < len(prob[0][0]) - 1:
                        prompt += ' '
                prompt += '] ['
                for i in range(len(prob[0][1])):
                    prompt += str(prob[0][1][i])
                    if i < len(prob[0][1]) - 1:
                        prompt += ' '
                prompt += ']\n['
                for i in range(len(prob[1][0])):
                    prompt += str(prob[1][0][i])
                    if i < len(prob[1][0]) - 1:
                        prompt += ' '
                prompt += ']'

            # Request payload
            data = {
                "model": "claude-2",  # Choose model (claude-2 or claude-instant)
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 300,
                "temperature": 0.7
            }

            # Make API call
            response = None
            while response is None:
                try:
                    api_response = requests.post(anthropic_endpoint, headers=headers, json=data)
                    if api_response.status_code == 200:
                        response = api_response.json().get('completion', '').strip()
                    else:
                        print(f"Error: {api_response.status_code}, {api_response.text}")
                        time.sleep(5)
                except Exception as e:
                    print(f"API request failed: {e}")
                    time.sleep(5)

            prob_type_responses.append(response)
        all_prob_type_responses.append(prob_type_responses)

    response_dict[alph] = all_prob_type_responses

    # Save results
    path = f'claude_pred/{args.gen}'
    check_path(path)
    save_fname = f'./{path}/claude_letterstring_results_{args.num_permuted}_multi_alph_probs'
    if args.promptstyle:
        save_fname += f'_{args.promptstyle}'
    if args.sentence:
        save_fname += '_sentence'
    if args.noprompt:
        save_fname += '_noprompt'
    save_fname += '.npz'
    np.savez(save_fname, all_prob_type_responses=response_dict, allow_pickle=True)