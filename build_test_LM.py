#!/usr/bin/python
import re
import nltk
import sys
import getopt
import itertools
import operator
import math
import numpy as np
from string import digits, punctuation

### CONFIGURATION PARAMETERS ########################################################
token_size = 4                      # How many characters are there in each token?
PADDING = True                      # Should there be START and END padding included?
INCLUDE_UPPERCASE = True            # Should the strings be converted to lowercase?
INCLUDE_NUMBERS = True              # Should numbers be included?
INCLUDE_PUNCTUATION = True          # Should punctuation (any non-alphanumeric character) be included?

smoothing_value = 1

# Set a threshold from 0 to 1.
# Sentences will be classified as "other" if they exceed this percentage of unclassified tokens.
other_threshold = 0.42               
#####################################################################################

### CONSTANTS
start_token = "START"
end_token = "END"
other_language = "other"

### A file reader for this language model. Reads the file by line.
###
def read_by_line(in_file):
    with open(in_file, 'r', encoding='utf-8') as f:
        return f.read().splitlines()

def tokenise(s, n):
    if PADDING:
        return tokenise_padding(s, n)
    else:
        return tokenise_no_padding(s, n)    

### A tokeniser method that takes in an iterable string of length s,
    # and generates a list of tokens of length n. Padding included.
###
def tokenise_padding(s, n):
    token_list = list()
    for i in range(1 - n, len(s)):      # Create these number of tokens
        current_token = list()

        for j in range (i, i + n):      # Tokens are of this length
            if j < 0:
                current_token.append(start_token)
            elif j >= len(s):
                current_token.append(end_token)
            else:
                current_token.append(s[j])

        token_list.append(tuple(current_token))     # Converted to tuple for hashing
        
    return token_list

### A tokeniser method that takes in an iterable string of length s,
    # and generates a list of tokens of length n. Padding not included.
###
def tokenise_no_padding(s, n):
    token_list = list()
    for i in range(len(s) - n + 1):         # Create these number of tokens
        current_token = tuple(s[i: i + n])  # Converted to tuple for hashing
        token_list.append(current_token)             
    return token_list

### A helper method to strip a string of the given characters.
###
def remove_from(s, unwanted):
    return s.translate({ord(k): None for k in unwanted})

### A parser that splits one training data into its label and list of tokens.
###
def parse_training_sentence(s):
    label = s.split()[0]
    string = s[len(label) + 1: ]

    if not INCLUDE_UPPERCASE:
        string = string.lower()

    if not INCLUDE_NUMBERS:
        string = remove_from(string, digits)

    if not INCLUDE_PUNCTUATION:
        string = remove_from(string, punctuation)

    tokens = tokenise(string, token_size)
    return label, tokens

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print('building language models...')
    
    data = read_by_line(in_file)

    training_data = [parse_training_sentence(d) for d in data]
    languages = set(d[0] for d in training_data)
    # Combine all the tokens and filter for unique tokens to form the model vocabulary
    vocabulary = set(itertools.chain.from_iterable(d[1] for d in training_data))

    # Create empty counter with smoothed value for all languages
    basic_counter = dict()
    for l in languages:
        basic_counter[l] = smoothing_value

    # Set up the model with vocabulary and basic counter for each token
    model = dict()
    for v in vocabulary:
        model[v] = basic_counter.copy()

    # Add the counts from the training data to the model
    for label, tokens in training_data:
        for t in tokens:
            model[t][label] += 1

    # Convert the count-based model to a probabilistic one
    counts = map(lambda l: sum(counts[l] for counts in model.values()), languages)
    language_counts = dict(zip(languages, counts))

    for count in model.values():
        for l in count:
            count[l] = math.log(count[l]/language_counts[l])

    return model, languages

def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")

    # Parse inputs
    data = read_by_line(in_file)
    model = LM[0]
    languages = LM[1]

    # Create an empty probability reference sheet
    basic_probability_ref = dict()
    for l in languages:
        basic_probability_ref[l] = np.float64(1)

    # Clear file of any results, to prepare the output file for eval.py
    open(out_file, 'w').close()
    
    with open(out_file, 'a') as f:
        # Process each sentence and make a guess
        for sentence in data:
            p_ref = basic_probability_ref.copy()
            tokens = tokenise(sentence, token_size)
            num_undetected = 0

            # Multiple token probabilities for each token
            for t in tokens:
                if t in model:
                    # Multiply the respective probability of each language
                    counts = model[t]
                    for l in languages:
                        p_ref[l] += counts[l]
                else:
                    num_undetected += 1

            # Find the best prediction using the maximum product
            if num_undetected / len(tokens) > other_threshold:
                prediction = other_language
            else:
                prediction = max(p_ref.items(), key=operator.itemgetter(1))[0]

            # Write the prediction to the output file
            f.write(prediction + " " + sentence + "\n")

def usage():
    print("usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file")

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-b':
        input_file_b = a
    elif o == '-t':
        input_file_t = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
