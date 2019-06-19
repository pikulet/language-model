This is the README file for
JOYCE YEO SHUHUI

== Python Version ==

I'm using Python Version <3.6> for this assignment.

== General Notes about this assignment ==

# Algorithm for Building the LM #

1. Parsing and Tokenising
To build the language model, I first parse all the input strings. The
parser recognises the language label, and treats the remaining sentence.
The treated string is then passed to the tokeniser to split into
n-length tokens.

2. Defining vocabulary and language labels
The next step is to define the scope (or vocabulary) of the language
model. This vocabulary space is the set of all tokens in the input. A
set is used to quickly handle duplicate entries. 

On top of defining the vocabulary of the tokens, I also defined the
types of languages in the input ("malaysian", "indonesian", "tamil")
by looking at the set of the labels. By avoiding the explicit
definition of these languages, I make the model easily extensible to
other input files with more language types.

3. Defining the model structure
With the set of language labels and the vocabulary defined, the model
can be put together. The model is essentially a dictionary of
dictionaries, that is structured like this:

model --> [key: token] --> [key: language] --> count (+ smoothing)

For example, looking up "emua" would give me a dictionary of the
respective counts for "emua" in the various languages. To retrieve the
count for a specific language, e.g. "indonesian", we look up the key
again. A sample search would then behave as follows:

model["emua"]["indonesian"] would give us the count value for "emua"
in Indonesian, including smoothing.

I chose to have the token as the first key (instead of the language)
to make the testing and prediction process more efficient. Imagine the
program reading the test string, it would look at the first token, and
look up the probabilities in the different languages for that token.
On the other hand, having the language as the first key would require
 more look-ups. (3 look-ups for the languages, which is the same as
before, but also 3 look-ups for the token, once for each language.
Earlier, we only have to look-up the token once regardless of the
number of languages in the model.)

4. Adding the smoothing and token counts
To exploit the similarities in the model, I chose to smooth the model
first, before adding the token counts. This allows me to give every
token in the model an initial smoothed value.

Every token's default would then be something like this:
[token] --> { 	"indonesian"	: 1,
		"malaysian" 	: 1,
		"tamil"		: 1 }

I then iterate through all the inputs and add the respective counts to
the model.

5. Conversion to a probabilistic model
A probabilistic model takes into account the length of the input
strings, providing a more accurate representation of the
statistical structure of the languages we are identifying.

Given the count values, I generated the total count for each
language once, then iterated through all the counts in the
language model by the total token count of the respective
languages.

Since I was iterating through all the counts in the model, I
leveraged on the iteration to apply a logarithmic function to all
the probabilities calculated. The logarithm will help prevent
data underflow, and also does not affect the predictive results.

Given probabilities A, B and C,
P(A) x P(B) > P(C) implies that log(A) + log(B) > log (C).
Note that because the logarithm was applied, I used addition to
treat the probability chain.

## Token Structure ##

### Why I chose to add padding? ###
The input strings are linear structures giving us a glimpse of the
language structures. Padding adds more information on the sentence
structures. For me, I explored the differences in the predictive
results with and without padding, and found no observerable
difference. I left the padding setting as a choice for users to
experiment with different model and string parsing techniques.

### Choice of Data Structures ###
An interesting point to bring out would be data immutability in
python. If we were to examine the structure of the tokens, they would
not be as simple as "emua", but instead be ('e', 'm', 'u', 'a').
The splitting of the strings into individual characters allows us to
easily insert the start and end tokens.

For example, ('<START>', 'e', 'm', 'u') can be done without having to
reserve special characters for the start and end tokens. This
decision to format the tokens as a tuple makes the model more
foolproof against special symbols in the input file.

Initially, I picked a list to represent the tokens, 
e.g. ['e', 'm', 'u', 'a']. However, the dictionary structure does not
permit the hashed keys to be a mutable data structure. I simply
overcame this by converting the mutable list to an immutable tuple,
though this incurs additional running time.

## Configuration Parameters ##

An interesting tool I incorporated into my project is to allow the
user to configure the tokenising parameters in the .py file. 

The changes include:
. Whether to pad the string with <START> and <END> tokens
. The length of each token (The assignment's default is 4.)
. String conversion to lowercase, stripping punctuation

Besides the tokenisation, the user can also configure build and test
parameters such as:
. The smoothing value in the model
. The threshold to classify test strings as "other" languages

# Algorithm for Testing the LM #

1. Tokenise the test strings
I tokenise the test strings using the same tokeniser as the input
string, for consistency. That is, if the input strings were stripped
of spaces, then the same would apply to the test strings.

2. Add logarithms
For every test string, I maintain a current probability tracker for
each language. Since I start off with logarithms, the base value is 0.
{ 	"indonesian"	: 0,
  	"malaysian" 	: 0,
 	"tamil"		: 0  }

For every token in the test string, I look up its respective count
in the language model. Referencing the same structure above, every
token would yield a dictionary similar to the one below:
[token] --> { 	"indonesian"	: 0.002,
		"malaysian" 	: 0.012,
		"tamil"		: 0.0003 }
For every language, I then add the logarithmic probability to the
probability tracker above.

2b. Treating invalid tokens
Since the training data is incomplete, there will be tokens which
do not exist in the language model. In this case, I ignore the token
and let it have no effect on my probability tracker. However, I add
a count of 1 to a separate invalid token counter. This counter tracks
the number of invalid tokens in the input string.

3. Making a prediction
For every test string, we can look at the sum of the logarithmic
probabilities in the probability tracker. We take the language with
the highest sum (done using an itemgetter), and this is the language
that the test string is most likely to be classified as, among the
other languages supported by the model.

3b. Classifying test strings as "other"
Before we make a definite prediction, I looked at the percentage of
tokens in the test string that were invalid. If this percentage
exceeded a certain threshold, i.e. more than x% of the tokens do not
exist in the model built on the supplied languages, then I conclude
that the test string belongs to neither of the languages supported by
the model.

4. The process is repeated for each of the test strings, so all the
test strings have predictions that are independent of each other.

# Deciding on the "Other" Threshold #

One key point of interest in this assignment is classifying test
strings that are not of the given languages, say an English sentence.

Based on the current implementation of testing there are two quick
metrics we can turn to, each with its own set of pros and cons.

1. Percentage of tokens in test string that are not in the model
I used this method in my prediction model. Suppose a sentence has x%
of its tokens that are not found in the vocabulary space of the model.
When x exceeds an arbitrarily determined threshold, my model will
mark the test string as belonging to neither of the three languages.

The main limitation of this method is that we are using what is not
present to make a characterise the test string.

2. Dominance of language prediction in comparison to other languages
One way to use only the tokens present to classify the test strings,
is to examine the probability dominance of the languages.

For each test string, we look at the probabilities across the
different languages. If one language particularly dominates, then it
could signify a non-random relationship in the test string. In the
scenario where all the languages have similar percentages, a conclusion
cannot be confidently made about the test string.

Again, the main issue with this method is that all languages can have
low probabilities (0.001%) vs (0.00001%) but we still identify the
language dominance. The auxiliary issue is language-specific: the
similarities between malaysian and indonesian are not well understood,
and could cause this method to fail.

After considering these two methods, I decided to implement the first:
a method that is cleaner and more straightforward.

## Tweaking the Threshold Value ##
Besides deciding on what metric to use, it is also imperative to set
the threshold value regardless of the method used.

For me, I picked the threshold by (1) tweaking around with different
token lengths and (2) testing with "other" strings

(1) In general, a language ngram with larger n will mean that fewer
tokens will match exactly with the training corpora, resulting in
a higher percentage of tokens which do not exist in the model. Then,
the value of the threshold will increase.

Given this understanding, I worked to find out the lower and upper
bound for the threshold, which still pass the 20 given test cases.
When the threshold is too low, real strings are classified as
"other" languages. When the threshold is too high, "other" strings
are given one of the three classifications. I compared the lower and
upper bounds across ngrams of different token lengths (3, 4, 5, 6)
to understand the resistance and significance of the threshold value.
Generally, the threshold value falls in the 30% - 50% range.

I played around to finally settle with 0.42. That is, when more than
42% of the tokens in the test string do not appear in the actual
language model, the test string is likely to be of "other" language.

(2) Testing with "other" strings
A more straightforward way is to actually put strings that are neither
of malaysian, indonesian or tamil, and observe how the model treats
them. The expected result is that the model should classify them as
"other".

== Files included with this submission ==

# build_test_LM.py
# eval.py

== Statement of individual work ==

[X] I, JOYCE YEO SHUHUI, certify that I have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, I
expressly vow that I have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions. 

== References ==

# Stripping numbers from string
	(https://stackoverflow.com/questions/12851791/
	removing-numbers-from-string/12856384)
# Using logarithms to circumvent the issue of probabilities being too
	small (SOH JASON)
