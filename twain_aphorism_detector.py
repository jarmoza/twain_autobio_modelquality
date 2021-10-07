# Author: Jonathan Armoza

# Let's detect some aphorisms, baby

# Imports

# Built-ins
import csv
import json
import os
import statistics

# Third party
import gensim
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression.rolling import RollingWLS
from tqdm import tqdm



# Globals


debug_flag = True
debug_separator = "========================================================================"
paths = {
	
	"aphorisms": "{0}{1}input{1}twain_aphorisms.csv".format(os.getcwd(), os.sep),
	"autobio_folder": "{0}{1}output{1}twain_autobio{1}".format(os.getcwd(), os.sep),
	"autobio_model": "{0}{1}input{1}twain_autobio.model".format(os.getcwd(), os.sep),
	"distances": "{0}{1}output{1}twain_distances.json".format(os.getcwd(), os.sep),
	"closest_sentences": "{0}{1}output{1}twain_closest_sentences.csv".format(os.getcwd(), os.sep)
}


# Utility functions

# From: https://medium.com/analytics-vidhya/linear-algebra-from-strang-3394007ec79c
def calc_proj_matrix(A):
    return A*np.linalg.inv(A.T*A)*A.T
def calc_proj(b, A):
    P = calc_proj_matrix(A)
    return P*b.T

def create_matrix_from_sentence(p_word_list, p_model):

	# print("Word list: {0}".format(p_word_list))

	# 1. Gather all word vectors of the word list in the word2vec model
	word_vectors = get_word_vectors_from_wordlist(p_word_list, p_model)

	# 2. Create a matrix of word vectors
	new_matrix = np.matrix(word_vectors)

	# print(new_matrix.shape)

	return new_matrix

def distance_from_sentence_to_sentence(p_sent1_wordlist, p_sent2_wordlist, p_model):

	# 1. Create a matrix and separate word vector list for the first sentence wordlist
	sent1_vectors = get_word_vectors_from_wordlist(p_sent1_wordlist, p_model)
	sent1_matrix = create_matrix_from_sentence(p_sent1_wordlist, p_model)

	# 2. Get all word vectors for second sentence
	sent2_vectors = get_word_vectors_from_wordlist(p_sent2_wordlist, p_model)

	# 3. Calculate projection vectors for each word in second sentence to
	# subspace made by first sentence matrix
	# proj_vectors = [calc_proj(v, sent1_matrix) for v in sent2_vectors]

	# From https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
	# The projection of a point q = (x, y, z) onto a plane given by a point p = (a, b, c) and a (unit) normal n = (d, e, f) is
	# q_proj = q - dot(q - p, n) * n
	n = np.linalg.norm(sent1_matrix)
	unit_n = n / np.linalg.norm(n)
	proj_vectors = [v - np.dot(v - sent1_matrix, unit_n) * unit_n for v in sent2_vectors]

	# print("Projection vectors: {0}\n\nSentence 2 vectors: {1}".format(proj_vectors, sent2_vectors))

	# 4. Create a distance vector between projected points on the first sentence plane and the second sentences' words
	distances = [np.linalg.norm(proj_vectors[i] - sent2_vectors[i]) for i in range(len(sent2_vectors))]
	
	# print("Distance from sentence words to aphorism plane:")
	# print(distances)
	# print("Norm of those distances:")
	# print(np.linalg.norm(distances))

	# 5. Return the magnitude of the distance vector
	return np.linalg.norm(distances)

def get_word_vectors_from_wordlist(p_word_list, p_model): 

	# print("Word list: {0}".format(p_word_list))

	# 1. Gather all word vectors of the word list in the word2vec model
	word_vectors = []
	for index in range(len(p_word_list)):

		# I. Make sure the aphorism word is in the model before retrieving a vector
		if p_word_list[index] in p_model.wv.key_to_index:

			# print("Storing vector for {0}: {1}".format(p_word_list[index], p_model.wv[p_word_list[index]]))
			word_vectors.append(p_model.wv[p_word_list[index]])

	return word_vectors


def main():

	if debug_flag:
		print("Reading in Twain aphorisms...")

	# 1. Read in aphorisms
	# https://www.aphorism4all.com/by_author.php?aut_id=542&p=0
	aphorisms = []
	with open(paths["aphorisms"], "r") as input_file:
		csv_reader = csv.reader(input_file)
		for row in csv_reader:

			# A. Preprocess each aphorism as a list of words (minus gensim stopwords)
			word_list = gensim.utils.simple_preprocess(row[0])

			# B. Save each aphorism word list
			aphorisms.append(word_list)


	if debug_flag:
		print("Reading in Twain autobiography...")

	# 2. Read in autobiography corpus (paragraphs)
	twain_volume_doc_count = { "1": 87, "2": 105, "3": 94 }
	twain_docs_raw = { "1": [], "2": [], "3": [] }
	twain_docs_bysent = { "1": [], "2": [], "3": [] }
	for volume_number in range(3):

		str_volume = str(volume_number + 1)
		for doc_number in range(twain_volume_doc_count[str_volume]):
		
			# A. Read in each document in volume and document order
			with open(paths["autobio_folder"] + "twain_autobio_vol{0}_body_{1}.txt".format(str_volume, doc_number)) as doc_file:

				# I. Read each document
				doc_data = doc_file.read()

				# II. Store each document in order (by volume number)
				twain_docs_raw[str_volume].append(doc_data)

				# III. Store a sentence tokenized version of each document in order (by volume number)
				twain_docs_bysent[str_volume].append(sent_tokenize(doc_data))


	# 3. Model autobiography corpus in word2vec
	if not os.path.exists(paths["autobio_model"]):

		if debug_flag:
			print("Modeling Twain autobiography with word2vec...")

		# A. Store docs in comprehensive list for modeling
		all_twain_docs = []
		for volume_number in range(3):
			all_twain_docs.extend(twain_docs_raw[str(volume_number + 1)])

		# B. Perform preprocessing on docs for gensim
		all_twain_docs = [gensim.utils.simple_preprocess(doc) for doc in all_twain_docs]

		# C. Train the model
		autobio_model = gensim.models.Word2Vec(sentences=all_twain_docs, 
											   vector_size=150, 
											   window=10, 
											   min_count=2, 
											   workers=10)

		if debug_flag:
			print("Saving Twain autobiography word2vec model to disk...")		

		# D. Save the model to disk
		autobio_model.save(paths["autobio_model"])

	else:

		if debug_flag:
			print("Loading Twain autobiography word2vec model from disk...")

		# A. Load the model from disk
		autobio_model = gensim.models.Word2Vec.load(paths["autobio_model"])


	# if debug_flag:
	# 	print("Creating word vector matrices for aphorisms...")

	# # 4. Produce word vector matrix for each aphorism based on autobio word2vec model
	# aphorism_matrices = []
	# for word_list in aphorisms:
			
	# 	# A. Create and store a matrix from the word vectors
	# 	aphorism_matrices.append(create_matrix_from_sentence(word_list, autobio_model))


	# 5. Consider each sentence as a word vector matrix and calculate its distance from each aphorism
	# and average those distances
	twain_doc_aphdist_bysent = { "1": [], "2": [], "3": [] }

	if not os.path.exists(paths["distances"]):

		if debug_flag:
			print("Computing average distances for each document's sentences to aphorisms...")

		# A. Calculate distances between sentences and aphorisms
		for volume_number in range(3):

			str_volume = str(volume_number + 1)

			print("Calculating distances for volume {0} ...".format(str_volume))
			
			for doc_number in tqdm(range(twain_volume_doc_count[str_volume])):

				# A. Compute average distances between doc's sentences and aphorisms
				doc_avg_distances = []
				for sent in twain_docs_bysent[str_volume][doc_number]:

					# I. Get word list of this sentence
					sent_word_list = gensim.utils.simple_preprocess(sent)

					# II. Measure the distance between each aphorism and this sentence
					distances = []
					for aph_word_list in aphorisms:
						distances.append(distance_from_sentence_to_sentence(aph_word_list, sent_word_list, autobio_model))

					# III. Compute and store the average distance
					doc_avg_distances.append(float(statistics.mean(distances)))

					# print("Doc avg distances:")
					# print(doc_avg_distances)

				# B. Save average distances from aphorisms for this doc
				twain_doc_aphdist_bysent[str_volume].append(doc_avg_distances)

		# B. Write distances and sentences to file here
		with open(paths["distances"], "w") as distance_file:

			json.dump(twain_doc_aphdist_bysent, distance_file)			

		# # A. Calculate distances between sentences and aphorisms
		# for volume_number in range(3):

		# 	print("Calculating distances for volume {0} ...".format(str_volume))

		# 	str_volume = str(volume_number + 1)
		# 	for doc_number in tqdm(range(twain_volume_doc_count[str_volume])):

		# 		# A. Compute average distances between doc's sentences and aphorisms
		# 		doc_avg_distances = []
		# 		for sent in twain_docs_bysent[str_volume][doc_number]:

		# 			# I. Create a matrix of the word vectors of this sentence
		# 			sent_matrix = create_matrix_from_sentence(gensim.utils.simple_preprocess(sent), autobio_model)

		# 			# print("Sentence matrix")
		# 			# print(sent_matrix)
		# 			# print(debug_separator)
		# 			# print("Aphorism matrices")
		# 			# print(aphorism_matrices)

		# 			# II. Check distance to each aphorism matrix
		# 			# def distance_fn(matrix1, matrix2):
		# 			# 	dist = (matrix1 - matrix2)**2
		# 			# 	dist = np.sum(dist, axis=1)
		# 			# 	return np.sqrt(dist)

		# 			distances = []
		# 			for aph_matrix in aphorism_matrices:

		# 				# A. Copy sentence matrix into list that matches aphorism matrix size
		# 				sent_matrix_list = sent_matrix.tolist()
		# 				new_sent_matrix_list = [[0] * aph_matrix.shape[1]] * aph_matrix.shape[0]
		# 				for index in range(sent_matrix.shape[0]):
		# 					for index2 in range(sent_matrix.shape[1]):
		# 						new_sent_matrix_list = sent_matrix_list[index][index2]

		# 				# B. Convert the new sentence 2d list into matrix
		# 				new_sent_matrix = np.matrix(new_sent_matrix_list)

		# 				# reshaped_sent_matrix = np.empty(aph_matrix.shape, dtype=np.ndarray)
		# 				# print(sent_matrix.shape)
						
		# 				# if sent_matrix.shape[0] < aph_matrix.shape[0]:
		# 				# for i in range(aph_matrix.shape[0]):
		# 				# 	for j in range(aph_matrix.shape[1]):
		# 				# 		print("sent_matrix[i,j]: {0},{1}".format(sent_matrix[i,j]))
		# 						# reshaped_sent_matrix[i][j] = sent_matrix[i][j]
							
		# 				# print(reshaped_sent_matrix.shape)
		# 				distances.append(np.linalg.norm(aph_matrix - new_sent_matrix))

		# 			# distances = [distance_fn(sent_matrix, aph_matrix) for aph_matrix in aphorism_matrices]

		# 			# III. Compute and store the average distance
		# 			doc_avg_distances.append(statistics.mean(distances))

		# 		# B. Save average distances from aphorisms for this doc
		# 		twain_doc_aphdist_bysent[str_volume].append(doc_avg_distances)

		# # B. Write distances and sentences to file here
		# with open(paths["distances"], "w") as distance_file:
		# 	json.dump(twain_doc_aphdist_bysent, distance_file)

	else:

		if debug_flag:
			print("Loading average distances for each document's sentences to aphorisms from disk...")

		with open(paths["distances"], "r") as distance_file:
			twain_doc_aphdist_bysent = json.load(distance_file)

	if debug_flag:
		print("Finding sentences with the smallest average distance to the aphorisms...")

	# 6. Take the sentence with the smallest distance from each document
	closest_sentences = { "1": [], "2": [], "3": [] }
	for volume_number in range(3):

		str_volume = str(volume_number + 1)
		for doc_number in range(twain_volume_doc_count[str_volume]):

			# A. Save the list index with the smallest distance
			smallest_distance = min(twain_doc_aphdist_bysent[str_volume][doc_number])
			smallest_distance_index = np.argmin(twain_doc_aphdist_bysent[str_volume][doc_number])

			# B. Save the smallest distance and the sentence with the smallest distance
			closest_sentences[str_volume].append([smallest_distance, twain_docs_bysent[str_volume][doc_number][smallest_distance_index]])


	# 7. Score the sentences on an aphorism detector scale between [0, 1]

	# A. Find farthest sentence
	farthest_distance = 0
	for volume_number in range(3):

		str_volume = str(volume_number + 1)
		for doc_number in range(twain_volume_doc_count[str_volume]):
			if closest_sentences[str_volume][doc_number][0] > farthest_distance:
				farthest_distance = closest_sentences[str_volume][doc_number][0]

	# B. Add new score on scale [0,1] to closest sentences data and prepare for scores plotting in autiobio order
	autobio_doc_index = 0
	for volume_number in range(3):

		str_volume = str(volume_number + 1)
		for doc_number in range(twain_volume_doc_count[str_volume]):

			# I. Save the aphorism distance for this sentence in the [0,1] scale
			aphorism_score = closest_sentences[str_volume][doc_number][0] / float(farthest_distance)
			closest_sentences[str_volume][doc_number].append(aphorism_score)

			# II. Record this doc's x-axis value (document number in autobio as a whole) for future plotting
			closest_sentences[str_volume][doc_number].append(autobio_doc_index)
			autobio_doc_index += 1

	if debug_flag:
		print("Writing closest sentences and their scores to disk...")

	# C. Output closest sentences to file with their aphorism scores
	with open(paths["closest_sentences"], "w") as output_file:

		output_file.write("score,sentence\n")

		for volume_number in range(3):
			str_volume = str(volume_number + 1)
			for doc_number in range(twain_volume_doc_count[str_volume]):
				output_file.write("{0},{1}\n".format(closest_sentences[str_volume][doc_number][0],
					closest_sentences[str_volume][doc_number][1]))

	if True:
		return

	# 8. Scatterplot the sentence data points (y - aphorism scale, x - document number)

	# A. X-Axis: Gather aphorism scores of closest sentences in volume,document order
	
	aphorism_scores = []
	for volume_number in range(3):
		str_volume = str(volume_number + 1)
		for doc_number in range(twain_volume_doc_count[str_volume]):
			
			# Regular scale
			# aphorism_scores.append(closest_sentences[str_volume][doc_number][0])

			# Aphorism scale
			aphorism_scores.append(closest_sentences[str_volume][doc_number][2])

	# B. Y-Axis document indices in overall autobiography
	doc_indices = list(range(autobio_doc_index))

	# I. Clear out zero aphorism scores and their indices
	cleaned_aphorism_scores = []
	cleaned_doc_indices = []
	for index in doc_indices:
		if aphorism_scores[index] > 0:
			cleaned_aphorism_scores.append(aphorism_scores[index])
			cleaned_doc_indices.append(index)

	# C. Do OLS regression for example
	df = pd.DataFrame({"X": cleaned_doc_indices, "Y": cleaned_aphorism_scores})
	df["bestfit"] = sm.OLS(df["Y"], sm.add_constant(df["X"])).fit().fittedvalues

	# D. Do a scatter plot of the aphorism scores over text time of the autobiography volumes
	# including a line for the OLS regression
	# fig = go.Figure(data=go.Scatter(x=cleaned_doc_indices, y=cleaned_aphorism_scores, mode="markers"))
	# fig.add_trace(go.Scatter(x=cleaned_doc_indices, y=df["bestfit"], mode="lines"))
	# fig.show()

	# if True:
	# 	return

	# 9. Run regression models over the data points, calculating the AIC score
	# Based on: https://www.statology.org/aic-in-python/ and https://www.statsmodels.org/stable/api.html

	# 1. Ordinary Least Squares
	# OLS(endog[, exog, missing, hasconst])

	# 2. Weighted Least Squares
	# WLS(endog, exog[, weights, missing, hasconst])

	# 3. Generalized Least Squares
	# GLS(endog, exog[, sigma, missing, hasconst])

	# 4. Generalized Least Squares with AR covariance structure
	# GLSAR(endog[, exog, rho, missing, hasconst])

	# 5. Recursive least squares
	# RecursiveLS(endog, exog[, constraints])

	# 6. Rolling Ordinary Least Squares
	# RollingOLS(endog, exog[, window, min_nobs, …])

	# 7. Rolling Weighted Least Squares
	# RollingWLS(endog, exog[, window, weights, …])

	regression_models = [
		sm.OLS,
		sm.WLS,
		sm.GLS,
		sm.GLSAR,
		sm.RecursiveLS,
		# RollingOLS,
		# RollingWLS,
	]
	# reg_model_names = ["OLS", "WLS", "GLS", "GLSAR", "RecursiveLS", "RollingOLS", "RollingWLS"]
	reg_model_names = ["OLS", "WLS", "GLS", "GLSAR", "RecursiveLS"]
	aic_results = [model(cleaned_aphorism_scores, cleaned_doc_indices).fit().aic for model in regression_models]

	# 10. Create a bar plot of the AIC scores for each regression model
	fig = go.Figure([go.Bar(x=reg_model_names, y=aic_results)])
	fig.show()

	# # 11. Compare OLS to GLSAR
	# # C. Do OLS regression for example
	# df = pd.DataFrame({"X": cleaned_doc_indices, "Y": cleaned_aphorism_scores})
	# df["nextbestfit"] = sm.OLS(df["Y"], sm.add_constant(df["X"])).fit().fittedvalues
	# df["bestfit"] = sm.GLSAR(df["Y"], sm.add_constant(df["X"])).fit().fittedvalues

	# # D. Do a scatter plot of the aphorism scores over text time of the autobiography volumes
	# # including a line for the OLS regression
	# fig = go.Figure(data=go.Scatter(name="Score", x=cleaned_doc_indices, y=cleaned_aphorism_scores, mode="markers"))
	# fig.add_trace(go.Scatter(name="OLS", x=cleaned_doc_indices, y=df["nextbestfit"], mode="lines"))
	# fig.add_trace(go.Scatter(name="GLSAR", x=cleaned_doc_indices, y=df["bestfit"], mode="lines"))
	# fig.show()	

if "__main__" == __name__:
	main()