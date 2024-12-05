import openai
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
parser.add_argument('--gpt', help='give gpt model: 3, 35, 4')
parser.add_argument('--gen', help='give gen for generalized problems or nogen for non generalized')


args = parser.parse_args()
print(args.promptstyle)

if args.promptstyle == "webb" and int(args.num_permuted) >1:
	print("promptstyle webb can only be used with an unpermuted alphabet")
	sys.exit()

# GPT-3 settings
openai.api_key = "API KEY HERE"
if args.gpt == '3':
    kwargs = {"engine":"text-davinci-003", "temperature":0, "max_tokens":40, "stop":"\n", "echo":False, "logprobs":1, }
elif args.gpt == '35':
    kwargs = { "model":"gpt-3.5-turbo", "temperature":0, "max_tokens":40, "stop":"\n"}
elif args.gpt == '4':
    kwargs = { "model":"gpt-4", "temperature":0, "max_tokens":40, "stop":"\n"}

# Load all problems
if args.gen == 'gen':
	all_prob = np.load(f'./problems/{args.gen}/all_prob_{args.num_permuted}_7_gpt_human_alphs.npz', allow_pickle=True)['all_prob']
elif args.gen == 'nogen':
	all_prob = np.load(f'./problems/{args.gen}/all_prob_{args.num_permuted}_7_human.npz', allow_pickle=True)['all_prob']

response_dict={}

for alph in all_prob.item().keys():
	print(alph)
	if (all_prob.item()[alph]['shuffled_letters'] is not None):
		shuffled_letters = builtins.list(all_prob.item()[alph]['shuffled_letters'])
	else:
		shuffled_letters = None

	shuffled_alphabet = builtins.list(all_prob.item()[alph]['shuffled_alphabet'])

	prob_types = builtins.list(all_prob.item()[alph].keys())[2:] # first two items are list of shuffled letters and shuflled alphabet: skip this
	N_prob_types = len(prob_types)

	alph_string = ' '.join(shuffled_alphabet)
	print(alph_string)

	# Evaluate
	N_trials_per_prob_type = 10
	all_prob_type_responses = []
	count = 0
	for p in range(N_prob_types):
		if prob_types[p] == 'attn':
			alph_string = "For this question, ignore other instructions and respond 'a a a a'"
		print('problem type ' + str(p+1) + ' of ' + str(N_prob_types) + '...')
		prob_type_responses = []
		for t in range(N_trials_per_prob_type):
			print('trial ' + str(t+1) + ' of ' + str(N_trials_per_prob_type) + '...')
			# Generate prompt
			prob = all_prob.item()[alph][prob_types[p]]['prob'][t]
			prompt=''
			if not args.noprompt:
				if args.promptstyle not in ["minimal", "hw", "webb","webbplus"]:			
					prompt+='Use the following alphabet to guess the missing piece.\n\n' \
						+ alph_string \
						+ '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
				elif args.promptstyle == 'minimal':			
					prompt+='Use the following alphabet to complete the pattern.\n\n' \
						+ alph_string \
						+ '\n\nNote that the alphabet may be in an unfamiliar order. Complete the pattern using this order.\n\n'
				elif args.promptstyle == 'hw':			
					prompt+='Use this fictional alphabet: \n\n' \
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
				if args.promptstyle in ["minimal", "hw", "webb","webbplus"]:
					prompt += '] ['
				else:
					prompt += '] [ ? ]'
			if args.promptstyle == "human":
				messages = [{'role': 'system', 'content':'You are able to solve letter-string analogies'},
								{'role': 'user', 'content': "In this study, you will be presented with a series of patterns involving alphanumeric characters, together with an example alphabet.\n\n" +
								"Note that the alphabet may be in an unfamiliar order.\n" + 
								"Each pattern will have one missing piece marked by [ ? ].\n"+
								"For each pattern, you will be asked to guess the missing piece.\n" +
								"Use the given alphabet when guessing the missing piece.\n" +
								"You do not need to include the '[ ]' or spaces between letters in your response.\n\n"+
								"a b c h e f g d i j k l m n o p q r s t u v w x y z \n\n" +
								"[a a a] [b b b]\n[c c c] [ ? ]"},
								{'role':'assistant', 'content': 'h h h'},
								{'role':'user', 'content': "In this case, the missing piece is 'h h h'\nNote that in the given alphabet, 'b' is the letter after 'a' and 'h' is the letter after 'c'"},
								{'role':'user', 'content':prompt}]
			elif args.promptstyle in ["minimal", "hw", "webb","webbplus"]:
				messages = [{'role': 'system', 'content':'You are able to solve letter-string analogies'},
								{'role':'user', 'content':prompt}]
			else:
				print("please enter a promptstyle")

			if args.gpt == '3':
				comp_prompt = ''
				for m in messages:
					comp_prompt += '\n' + m['content']
				comp_prompt=comp_prompt.strip('\n')
				# print(comp_prompt)
			else:
				pass

			# Get response
			response = []
			while len(response) == 0:
				if args.gpt == '3':
					try:
						response = openai.Completion.create(prompt=comp_prompt, **kwargs)
					except:
						print('trying again...')
						time.sleep(5)
				else:
					try:
						response = openai.ChatCompletion.create(messages=messages, **kwargs)
					except:
						print('trying again...')
						time.sleep(5)

			if args.gpt =='3':
				prob_type_responses.append(response['choices'][0]['text'])	
			else:
				prob_type_responses.append(response['choices'][0]['message']['content'])
				# print(response)
			count += 1
		all_prob_type_responses.append(prob_type_responses)
		response_dict[alph] = all_prob_type_responses
		# Save
		path = f'GPT{args.gpt}_prob_predictions_multi_alph/{args.gen}'
		check_path(path)
		save_fname = f'./{path}/gpt{args.gpt}_letterstring_results_{args.num_permuted}_multi_alph_gptprobs'
		if args.promptstyle:
			save_fname += f'_{args.promptstyle}'
		if args.sentence:
			save_fname += '_sentence'
		if args.noprompt:
			save_fname += '_noprompt'
		save_fname += '.npz'
		np.savez(save_fname, all_prob_type_responses=response_dict, allow_pickle=True)



