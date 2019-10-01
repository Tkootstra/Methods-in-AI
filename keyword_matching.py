import random
import re

# Get the dialog act that matches a key. A key can have spaces.
def word_to_dialogAct(key):
  # The array of dialog acts and the keys that match them respectively.
  knowledge = [
    ["ack",       ["okay", "well", "kay", "great", "thatll do", "that one", "im good"]],
    ["affirm",    ["yes", "right", "ye", "yeah", "correct", "uh huh", "yea"]],
    ["bye",       ["bye", "stop", "thats all"]],
    ["confirm",   ["is it", "does it", "is there"]],
    ["deny",      ["not", "wrong", "dont want"]],
    ["hello",     ["hello", "hi"]],
    ["inform",    ["any", "dont care",                          # Miscellaneous.
                   "moderate", "expensive", "cheap",            # Price types.
                   "west", "north", "south", "centre", "east",  # Area types.
                   "british", "modern european", "italian",     # Food types.
                   "romanian", "seafood", "chinese",
                   "steakhouse", "asian oriental", "french",
                   "portuguese", "indian", "spanish",
                   "european", "vietnamese", "korean", "thai",
                   "moroccan", "swiss", "fusion", "gastropub",
                   "tuscan", "international", "traditional",
                   "mediterranean", "polynesian", "african",
                   "turkish", "bistro", "north american",
                   "australasian", "persian", "jamaican",
                   "lebanese", "cuban", "japanese", "catalan"]],
    ["negate",    ["no"]],
    ["repeat",    ["repeat", "back", "again"]],
    ["reqalts",   ["what about", "how about", "anything else", "other", "is there anything", "is there another"]],
    ["reqmore",   ["more"]],
    ["request",   ["whats", "can i", "address", "phone number", "post code", "price range", "type"]],
    ["restart",   ["start over", "reset", "start again"]],
    ["thankyou",  ["thank"]],
  ]

  for row in knowledge:
    if (key in row[1]):
      dialog_act = row[0]
      return dialog_act

# Get the dialog acts that match multiple potential keywords.
def sentence_to_dialogAct(sentence):
  dialog_acts = []
  sentence = sentence.lower()
  words = re.findall(r'\w+', sentence)

  # Dialog acts of single words (singles).
  for word in words:
    dialog_acts.append(word_to_dialogAct(word))
  # Dialog acts of word pairs (doubles).
  if len(words) >= 2:
    for i in range(0, len(words) - 1):
      dialog_acts.append(word_to_dialogAct(words[i] + " " + words[i + 1]))
  # Dialog acts of word triplets (triples).
  if len(words) >= 3:
    for i in range(0, len(words) - 2):
      dialog_acts.append(word_to_dialogAct(words[i] + " " + words[i + 1] + " " + words[i + 2]))

  return dialog_acts

# This method returns a random match or "Null" when it finds none.
def getDialogAct_logic(sentence):
  dialog_acts = sentence_to_dialogAct(sentence)

  # Filter out the None's.
  without_nones = []
  for dialog_act in dialog_acts:
    if dialog_act is not None:
      without_nones.append(dialog_act)

  if (len(without_nones) == 0):
    return "Null"
  else:
    return random.choice(without_nones)

"""
# Classify user input sentences.
while True:
  print("Please enter a sentence for classification.")
  user_input = input()
  print("You entered: " + user_input)
  print("Classified as dialog act: " + getDialogAct_logic(user_input))
"""