import keyword_matching
import pandas as pd
import random
import pickle
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
import numpy as np



filename = 'finalized_model.sav'
ML_model = pickle.load(open(filename, 'rb'))

rest_info = pd.read_csv('restaurants.csv')
foodUniques = list(rest_info['food'].unique())
priceUniques = list(rest_info['pricerange'].unique())
areaUniques = list(rest_info['area'].unique())[:5]

stopwords = stop_words = list(set(stopwords.words('english')))
stop_words.append('want')


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
        return candidates[0]


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

def predictDialogAct(utterance, model):
    sentence = utterance.lower()
    vector = vectorizeSentence(sentence.split())
    prediction = model.predict(vector)
    return prediction



currentCheck = None
currentState = "welcome"
options = []
stop = False
user_utterance = None

# The knowledge of the restaurant(s). When the user expresses a preference, this is now a requirement.
decision = {
  "restaurantname": None,
  "pricerange": None,
  "area": None,
  "food": None,
  "phone": None,
  "addr": None,
  "postcode": None
}

# Methods for the checks as in the dialog diagram (diamond shape).
checks = {}
# Methods for the states as in the dialog diagram (rounded shape).
states = {}

# All possible utterances of the system.
utterances = {
  "anythingElse": "can I help you with anything else?",
  "reject": "There is no restaurant matching your preferences.",
  "shutdown": "Shutting down.",
  "suggest": "{restaurantname} is a nice restaurant in the {area} of town",
  "suggest_addition_food": " serving {food} food",
  "suggest_addition_pricerange": " in the {pricerange} price range",
  "welcome": "hello. Welcome to the Cambridge restaurant system. You can ask for restaurants by area, price range or food type. How may I help you?",
}

# Start up the system.
def initialize():
  while (True):
    iteration()

# Output an utterance to the user.
def output(template):
  global decision

  # Fill in the variables of the template.
  for key in decision:
    pattern = "{" + key + "}"
    if pattern in template:
      template = template.replace(pattern, decision[key])

  utterance = template
  print("SYSTEM: " + utterance)

# Main algorithm.
def iteration():
  global stop
  global user_utterance
  global utterances

  # Step 1: system state and output.
  state(currentState)
  if (stop == True):
    output(utterances["shutdown"])
    exit()

  # Step 2: receive and analyze the user utterance.
  user_utterance = input()
  analyze(user_utterance)

  # Step 3: perform a check to make a transition to the next state.
  checks[currentCheck]()

def analyze(utterance):
  global decision

  # TODO: analyze utterance and update preferences pricerange/area/food in global decision variable

# Get the perceived dialog act of the last user utterance.
def getDialogActMachineLearning(utterance):
    prediction = predictDialogAct(utterance, ML_model)
    return prediction

# Check: do we have enough knowledge of the user preferences?
def check_knowledge():
  global user_utterance
  global options

  transition = None

  if getDialogAct(user_utterance) == "bye":
   # The user asked to shut down the system.
   setState("end")
  else:
    updateOptions()
    if len(options) == 0:
      # There exists no restaurant given the current preferences.
      setState("reject")
    else:
      firstRow = {
        "pricerange": options[0]["pricerange"],
        "area": options[0]["area"],
        "food": options[0]["food"]
      }
      fullMatch = True
      for option in options:
        for preference in ["pricerange", "area", "food"]:
          if option[preference] != firstRow[preference]:
            fullMatch = False
      if (fullMatch == True):
        setState("suggest")
      else:
        setState("askPreferences")
checks["knowledge"] = check_knowledge

# Check: did the user just ask for restaurant details?
def check_detailsQuestion():
  global user_utterance

  dialogAct = getDialogAct(user_utterance)
  if dialogAct == "bye":
    # The user asked to shut down the system.
    setState("end")
  elif dialogAct == "request":
    # The user asked for information about the restaurant.
    setState("giveDetails")
  elif dialogAct == "Null":
    # The user was not understood.
    setState("anythingElse")
  else:
    # The user did not ask for information about the restaurant.
    check_knowledge()
checks["detailsQuestion"] = check_detailsQuestion

# Umbrella method for calling specific state methods.
def state(stateName):
  global utterances
  global decision

  states["state_" + stateName](utterances, decision) 

# Update the global state variable.
def setState(stateName):
  global currentState
  currentState = stateName

def setNextCheck(checkName):
  global currentCheck
  currentCheck = checkName

# Greet the user and mention the systems' functionality.
def state_welcome(utterances, decision):
  global currentCheck

  output(utterances["welcome"])

  setNextCheck("knowledge")
states["state_welcome"] = state_welcome

# Mention that there are no restaurants satisfying the current preferences.
def state_reject(utterances, decision):
  global currentCheck
  # TODO improve: variable rejections mentioning the current preferences. Example: "there is no vegetarian restaurant in the cheap price range"
  output(utterances["reject"])

  setNextCheck("knowledge")
states["state_reject"] = state_reject

# Suggest a random restaurant that satisfies the current preferences.
def state_suggest(utterances, decision):
  global currentCheck
  
# =============================================================================
# hier moet dus de classifier in om een dialog act te predicten
# =============================================================================
  
  
  choice = random.choice(options)
  for key in decision:
    decision[key] = choice[key]

  addition = "pricerange" if restaurant["pricerange"] != None else "food"
  output(utterances["suggest"] + utterances["suggest_" + addition])

  setNextCheck("detailsQuestion")
states["state_suggest"] = state_suggest

# State: ask the user for restaurant preferences.
def state_askPreferences(utterances, decision):
  global currentCheck

  output("TODO finish state_askPreferences: ask for preferences")

  setNextCheck("knowledge")
states["state_askPreferences"] = state_askPreferences

# State: give the user the restaurant details that were asked for.
def state_giveDetails(utterances, decision):
  global currentCheck

  output("TODO finish state_giveDetails: give restaurant detail")

  setNextCheck("detailsQuestion")
states["state_giveDetails"] = state_giveDetails

# State: ask if there is anything else to help with.
def state_anythingElse(utterances, decision):
  global currentCheck

  output(utterances["anythingElse"])

  setNextCheck("detailsQuestion")
states["state_anythingElse"] = state_anythingElse

# State: close the system.
def state_end(utterances, decision):
  global currentCheck
  global stop

  stop = True
states["state_end"] = state_end

# Filter restaurants on the current preferences.
def updateOptions():
  global options
  global decision

  options = []
  # Read the restaurants data and filter on what is possible given the current preferences.
  restaurants = pd.read_csv("restaurants.csv")
  for index, row in restaurants.iterrows():
    if ((decision["pricerange"] in [None, row["pricerange"]])
      and (decision["area"] in [None, row["area"]])
      and (decision["food"] in [None, row["food"]])):
        options.append(row)

initialize()