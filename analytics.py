# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:53:06 2019

@author: timok
"""

# =============================================================================
# imports
# =============================================================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import pickle 
import json
import Levenshtein as lev
from nltk.corpus import stopwords

# =============================================================================
# helper methods for bag of words
# =============================================================================

json_file = 'ontology_dstc2.json'

with open(json_file) as file:
    data = json.load(file)
    
def defineWordNumbers(wordList):
    uniques = list(set([word.lower() for word in wordList]))
    numbers = [x for x in range(len(uniques))]
    return uniques, numbers

def vectorizeSentence(wordlist):
    import numpy as np
    mask = np.zeros([len(words)], dtype=int)
    for word in wordlist:
        try:
            idx = words.index(word)
            mask[idx] = 1
        except ValueError:
            pass

    return mask

#random.randrange(len(wordMatches)))

# =============================================================================
# vectorize all known input words found in data
# =============================================================================
full_data = pd.read_csv('full_data_dialogact_content.csv')
full_data['utterance_content'] = full_data['utterance_content'].apply(lambda words: words.lower())

all_words = []

for sentence in list(full_data['utterance_content']):
    for word in sentence.split():
        all_words.append(word.lower())
        

    
words, numbers = defineWordNumbers(all_words)

all_features = np.empty([len(full_data), len(words)])

all_sentences = list(full_data['utterance_content'])
for i in range(len(all_sentences)):
    sentence = all_sentences[i]
    vector = vectorizeSentence(sentence.split())
    all_features[i,:] = vector

all_features = pd.DataFrame(all_features)
all_features['label'] = full_data['dialog_act']
all_features = all_features.dropna()
# =============================================================================
# load trained classifier model into memory for making predictions
# =============================================================================

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


# =============================================================================
# chatbot input - output
# =============================================================================

import pandas as pd
import json
rest_info = pd.read_csv('restaurants.csv')


    
foodUniques = list(rest_info['food'].unique())
priceUniques = list(rest_info['pricerange'].unique())
areaUniques = list(rest_info['area'].unique())[:5]
    


stopwords = stop_words = list(set(stopwords.words('english')))
stop_words.append('want')

def levSteinWordMatch(inputWord, possibleCandidates):
    
    levDistances = []
    for word in possibleCandidates:
#        print(word)
        dist = lev.distance(inputWord,word)
        levDistances.append(dist)
    if min(levDistances) >= 3:
        return None
    candidates = possibleCandidates[np.argmin(levDistances)]
    if type(candidates) == str:
        return candidates
    else:
        return str(np.random.choice(candidates, size=1)[0])


def matchKeyWord(sentence, keywords=list):
    preference = 'missing'
    if type(sentence) == str:
        allWords = [word.lower() for word in sentence.split()]
        allWords = [word for word in allWords if word not in stopwords]
    else:
        allWords = sentence
#    check normal keyword match
    for word in allWords:
        if word in keywords:
            preference = word
#   check levStein keyword match
    if preference == 'missing':
        for word in allWords:
            levSteinWord = levSteinWordMatch(word, keywords)
            if levSteinWord in keywords:
                preference = levSteinWord
    return preference

def matchPref(sentence, keywords=list):
    allWords = [word.lower() for word in sentence.split()]
    allWords = [word for word in allWords if word not in stopwords]
#    print(allWords)
    #    edge cases
    if 'world' in allWords and 'food' in sentence and 'international' in keywords:
        return 'international'
    
    preference = 'missing'
    for i in range(len(allWords)):
        word = allWords[i]
        if word in keywords:
            preference = word
    
    if preference == 'missing':
        preference = matchKeyWord(sentence, keywords)
        
    return preference

def matchPrefFood(sentence, keywords=list):
    allWords_copy = [word.lower() for word in sentence.split()]
    allWords = [word for word in allWords_copy if word not in stopwords]
    
# =============================================================================
#     some food preferences are composed of two words, handle those as edge cases
# =============================================================================
    for k in range(len(allWords)-1):
        bigram = allWords[k] + ' '  + allWords[k+1]
        for word in keywords:
            if bigram == word:
                try:
                    return word, allWords[k-1]
                except IndexError:
                    pass
            levSteinWord = levSteinWordMatch(bigram, keywords)
            if levSteinWord == word:
                try:
                    return word, allWords[k-1]
                except IndexError:
                    pass
# =============================================================================
#   normal cases
# =============================================================================
    
    inputPreference = 'missing'
    #    edge cases
    
    preference = 'missing'
    for i in range(len(allWords)):
        word = allWords[i]
        if word in keywords:
            preference = word
        if 'food' in word:
            inputPreference = allWords[i-1]
    
    if preference == 'missing':
        preference = matchKeyWord(sentence, keywords)
    if 'world' in allWords and 'food' in sentence and 'international' in keywords:
        return 'international', inputPreference
        
    return preference, inputPreference



def getInformation(sentence):
    foodPref, input_ = matchPrefFood(sentence, foodUniques)
    pricePref = matchPref(sentence, priceUniques)
    areaPref = matchPref(sentence, areaUniques)
    for pref in [foodPref, pricePref, areaPref]:
        if pref == 'missing':
            pref = None
    return pricePref, areaPref, foodPref
    


    
#    
#
#for sentence in testSentences:
#    print('test sentence: '+sentence)
#    foodPref, input_ = matchPrefFood(sentence, foodUniques)
#    pricePref = matchPref(sentence, priceUniques)
#    areaPref = matchPref(sentence, areaUniques)
#    if foodPref == 'missing':
#        print('no restaurants found for keyword : {}'.format(input_))
##    print(sentence)
#    print('food preference": {}, price preference: {}, area preference:{}'.format(foodPref, pricePref, areaPref))
#    print('. ')
#            
        
    
    





    
    
    



    


    