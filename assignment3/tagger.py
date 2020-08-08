import numpy as np

from util import accuracy
# from data_process import Line
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	num_state = len(tags)
	A = np.zeros((num_state, num_state))
	pi = np.zeros(num_state) 

	state_dict, obs_dict = {}, {}
	word_list = []

	for idx, tag in enumerate(tags):
		state_dict[tag] = idx

	# create a word corpus
	for cur_line in train_data:
		for idx in range(cur_line.length):
			word_list.append(cur_line.words[idx])

	# initialize observation dict
	word_list = list(set(word_list)) 
	num_obs = len(word_list)
	for idx, word in enumerate(word_list):
		obs_dict[word] = idx

	# compute emission prob B
	B = np.zeros([num_state, num_obs])
	for lines in train_data:
		# compute initial probability
		pi[state_dict[lines.tags[0]]] += 1
		for idx in range(len(lines.words) - 1):
			o1 = obs_dict[lines.words[idx]]
			s = state_dict[lines.tags[idx]]
			s_dash = state_dict[lines.tags[idx + 1]]
			# compute transition probability
			A[s, s_dash] += 1
			# compute emission probability
			B[s, o1] += 1
		B[state_dict[lines.tags[len(lines.words) - 1]], obs_dict[lines.words[len(lines.words) - 1]]] += 1

	# normalize probability
	for row in range(len(A)):
		A[row, :] = A[row, :] / sum(A[row, :])
	for row in range(len(B)):
		B[row, :] = B[row, :] / sum(B[row, :])
	pi = pi / sum(pi)

	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	N = len(model.pi) 
	idx = len(model.obs_dict)
	prob = np.ones((N, 1)) * 1e-6 # 6 digit significance
	for line in test_data:
		for i in range(len(line.words)):
			cur_word = line.words[i]
			if cur_word not in model.obs_dict:
				model.obs_dict[cur_word] = idx  # update obs dict
				model.B = np.append(model.B, prob, axis=1)
				idx += 1

	# Tag sentence with viterbi algorithm
	for line in test_data:
		new_path = model.viterbi(line.words)
		tagging.append(new_path)
	###################################################
	return tagging
