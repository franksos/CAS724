# Hidden Markov Models in Python
# Katrin Erk, March 2013 updated March 2016
#
# This HMM addresses the problem of part-of-speech tagging. It estimates
# the probability of a tag sequence for a given word sequence as follows:
#
# Say words = w1....wN
# and tags = t1..tN
#
# then
# P(tags | words) is_proportional_to  product P(ti | t{i-1}) P(wi | ti)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(tags | words)
import nltk
import sys
from nltk.corpus import brown

# Estimating P(wi | ti) from corpus data using Maximum Likelihood Estimation (MLE):
# P(wi | ti) = count(wi, ti) / count(ti)
#
# We add an artificial "start" tag at the beginning of each sentence, and
# We add an artificial "end" tag at the end of each sentence.
# So we start out with the brown tagged sentences,
# add the two artificial tags,
# and then make one long list of all the tag/word pairs.

brown_tags_words = [ ]
for sent in brown.tagged_sents():
    # sent is a list of word/tag pairs
    # add START/START at the beginning
    brown_tags_words.append( ("START", "START") )
    # then all the tag/word pairs for the word/tag pairs in the sentence.
    # shorten tags to 2 characters each
    brown_tags_words.extend([ (tag[:2], word) for (word, tag) in sent ])
    # then END/END
    brown_tags_words.append( ("END", "END") )

# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))

# Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
# P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
brown_tags = [tag for (tag, word) in brown_tags_words ]

# make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)

print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))


###
# putting things together:
# what is the probability of the tag sequence "PP VB TO VB" for the word sequence "I want to race"?
# It is
# P(START) * P(PP|START) * P(I | PP) *
#            P(VB | PP) * P(want | VB) *
#            P(TO | VB) * P(to | TO) *
#            P(VB | TO) * P(race | VB) *
#            P(END | VB)
#
# We leave aside P(START) for now.

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") * \
    cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") * \
    cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") * \
    cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is:", prob_tagsequence)

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("saw") * \
    cpd_tags["VB"].prob("PP") * cpd_tagwords["PP"].prob("her") * \
    cpd_tags["PP"].prob("NN") * cpd_tagwords["NN"].prob("duck") * \
    cpd_tags["NN"].prob("END")

print( "The probability of the tag sequence 'START PP VB PP NN END' for 'I saw her duck' is:", prob_tagsequence)

prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("saw") * \
    cpd_tags["VB"].prob("PP") * cpd_tagwords["PP"].prob("her") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("duck") * \
    cpd_tags["VB"].prob("END")

print( "The probability of the tag sequence 'START PP VB PP VB END' for 'I saw her duck' is:", prob_tagsequence)


#####
# Viterbi:
# If we have a word sequence, what is the best tag sequence?
#
# The method above lets us determine the probability for a single tag sequence.
# But in order to find the best tag sequence, we need the probability
# for _all_ tag sequence.
# What Viterbi gives us is just a good way of computing all those many probabilities
# as fast as possible.

# what is the list of all tags?
distinct_tags = set(brown_tags)

sentence = ["I", "want", "to", "race" ]
#sentence = ["I", "saw", "her", "duck" ]
sentlen = len(sentence)

# viterbi:
# for each step i in 1 .. sentlen,
# store a dictionary
# that maps each tag X
# to the probability of the best tag sequence of length i that ends in X
viterbi = [ ]

# backpointer:
# for each step i in 1..sentlen,
# store a dictionary
# that maps each tag X
# to the previous tag in the best tag sequence of length i that ends in X
backpointer = [ ]

first_viterbi = { }
first_backpointer = { }
for tag in distinct_tags:
    # don't record anything for the START tag
    if tag == "START": continue
    first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
    first_backpointer[ tag ] = "START"

print(first_viterbi)
print(first_backpointer)
    
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)

currbest = max(first_viterbi.keys(), key = lambda tag: first_viterbi[ tag ])
print( "Word", "'" + sentence[0] + "'", "current best two-tag sequence:", first_backpointer[ currbest], currbest)
# print( "Word", "'" + sentence[0] + "'", "current best tag:", currbest)

for wordindex in range(1, len(sentence)):
    this_viterbi = { }
    this_backpointer = { }
    prev_viterbi = viterbi[-1]
    
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue

        # if this tag is X and the current word is w, then 
        # find the previous tag Y such that
        # the best tag sequence that ends in X
        # actually ends in Y X
        # that is, the Y that maximizes
        # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        # The following command has the same notation
        # that you saw in the sorted() command.
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: \
            prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

        # Instead, we can also use the following longer code:
        # best_previous = None
        # best_prob = 0.0
        # for prevtag in distinct_tags:
        #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
        #    if prob > best_prob:
        #        best_previous= prevtag
        #        best_prob = prob
        #
        this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
            cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
        this_backpointer[ tag ] = best_previous

    currbest = max(this_viterbi.keys(), key = lambda tag: this_viterbi[ tag ])
    print( "Word", "'" + sentence[ wordindex] + "'", "current best two-tag sequence:", this_backpointer[ currbest], currbest)
    # print( "Word", "'" + sentence[ wordindex] + "'", "current best tag:", currbest)


    # done with all tags in this iteration
    # so store the current viterbi step
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)


# done with all words in the sentence.
# now find the probability of each tag
# to have "END" as the next tag,
# and use that to find the overall best sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# best tagsequence: we store this in reverse for now, will invert later
best_tagsequence = [ "END", best_previous ]
# invert the list of backpointers
backpointer.reverse()

# go backwards through the list of backpointers
# (or in this case forward, because we have inverter the backpointer list)
# in each case:
# the following best tag is the one listed under
# the backpointer for the current best tag
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()
print( "The sentence was:", end = " ")
for w in sentence: print( w, end = " ")
print("\n")
print( "The best tag sequence is:", end = " ")
for t in best_tagsequence: print (t, end = " ")
print("\n")
print( "The probability of the best tag sequence is:", prob_tagsequence)