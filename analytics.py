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

#def sampleWordContext()


    

    
#    minValue = min(listvalues)
#    if minValue>=3:
#        return("null")
#    
#    wordMatches = []
#    i = 0
#    for index in listvalues:
#        if index==minValue:
#           wordMatches.append(wordList[i])
#        i = i+1
#    
#    import random
#    return(wordMatches[random.randrange(len(wordMatches))])

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
   

#print('welcome to restaurant chatbot 0.1. How can I help you?')
#print('input inform type dialog act')
#currentUserInput = input()
#userInputVectorized = vectorizeSentence(currentUserInput.lower().split())
#dialogActPred = model.predict(userInputVectorized.reshape(1,-1))[0]
##if dialogActPred == 'inform':
#    
#    
#    
##    check alle keywords
    
testSentences =["I'm looking for world food",
    "I want a restaurant that serves world food",
    "I want a restaurant serving Swedish food,"
    "I'm looking for a restaurant in the center",
    "I would like a cheap restaurant in the west part of town",
    "I'm looking for a moderately priced restaurant in the west part of town",
    "I'm looking for a restaurant in any area that serves Tuscan food",
    "Can I have an expensive restaurant",
    "I'm looking for an expensive restaurant and it should serve international food",
    "I need a Cuban restaurant that is moderately priced",
    "I'm looking for a moderately priced restaurant with Catalan food",
    "What is a cheap restaurant in the south part of town",
    "What about Chinese food",
    "I wanna find a cheap restaurant",
    "I'm looking for Persian food please",
    "Find a Cuban restaurant in the center"]

stopwords = stop_words = list(set(stopwords.words('english')))
stop_words.append('want')
#def matchPref(sentence, keywords=list):
#    allWords = [word.lower() for word in sentence.split() if word.lower() not in stop_words]
#    print(allWords)
#    wordsToCheck = None
#    for i in range(len(allWords)):
#        word = allWords[i]
#        if word in keywords:
#            wordsToCheck = allWords[i-2:i+2]
#    print(wordsToCheck)
#    if wordsToCheck == None:
#        return None
#    preference = matchKeyWord(wordsToCheck, keywords)
#    return preference

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
    
    wordsToCheck = None
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
#    print(allWords)
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


for sentence in testSentences:
    print('test sentence: '+sentence)
    foodPref, input_ = matchPrefFood(sentence, foodUniques)
    pricePref = matchPref(sentence, priceUniques)
    areaPref = matchPref(sentence, areaUniques)
    if foodPref == 'missing':
        print('no restaurants found for keyword : {}'.format(input_))
#    print(sentence)
    print('food preference": {}, price preference: {}, area preference:{}'.format(foodPref, pricePref, areaPref))
    print('. ')
    

def getInformation(sentence):
    foodPref, input_ = matchPrefFood(sentence, foodUniques)
    pricePref = matchPref(sentence, priceUniques)
    areaPref = matchPref(sentence, areaUniques)
    for pref in [foodPref, pricePref, areaPref]:
        if pref == 'missing':
            pref = None
    return pricePref, areaPref, foodPref
    


    
    
        
        
    
    





    
    
    



    


    