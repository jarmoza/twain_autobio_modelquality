# Author: Jonathan Armoza
# Creation date: November 2019
# Purpose: Maintains commonly used string functions for Art of Literary Modeling

# Built-ins

import os   	   # Working directory and folder separator char
import string 	   # Punctuation
import unicodedata # Removing diacritics from characters

# Utility functions

def clean_string(p_original_string):

	# 1. Strip whitespace and lowercase
	new_str = p_original_string.strip().lower()

	# 2. Remove all accents
	new_str = remove_diacritics(new_str)

	# 3. Replace all \n and \t with ' '
	new_str = new_str.replace("\n", " ").replace("\t", " ")

	# 4. Remove punctuation
	new_str = remove_punctuation(new_str)

	# 5. Split by spaces
	new_str_parts = new_str.split()
	# a. Removing single n's - unicode error converted em-dash to n-tilda
	new_str_parts = [part for part in new_str_parts if "n" != part]

	# 6. Rejoin with single spaces
	new_str = " ".join(new_str_parts)

	return new_str

def format_path(p_original_filepath):

	new_path = p_original_filepath.strip()
	return new_path + os.sep if os.sep != new_path[len(new_path) - 1] else new_path

# Source: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
def remove_diacritics(p_original_string):

	new_str = []
	for char in p_original_string:
	    # Gets the base character of char, by "removing" any
	    # diacritics like accents or curls and strokes and the like.

	    desc = unicodedata.name(char)
	    cutoff = desc.find(" WITH ")
	    if cutoff != -1:
	        desc = desc[:cutoff]
	        try:
	            char = unicodedata.lookup(desc)
	        except KeyError:
	            continue  # removing "WITH ..." produced an invalid name
	    new_str.append(char)

	return "".join(new_str)

def remove_punctuation(p_original_string):

	new_str_parts = []
	for char in p_original_string:
		if char in string.punctuation:
			continue
		new_str_parts.append(char)
	return "".join(new_str_parts)
