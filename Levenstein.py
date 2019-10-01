# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:30:20 2019

@author: harme
"""

def wordMatch(inputstring, informableTypes):
    fileString = 'C:\\Users\\harme\\Documents\\Studie\\AI\Master\\Methods in AI research\\Code\\ontology_dstc2.json'
    import json
    with open(fileString) as jsonfile:
        data = json.load(jsonfile)
    
    wordList = []
    for types in informableTypes:
        for elements in data['informable'][types]:        
            wordList.append(elements)
    
    listvalues = []
    import Levenshtein as lev
    t = 0
    for words in wordList:    
        listvalues.append((lev.distance(inputstring.lower(),words.lower())))
        t = t+1
    
    minValue = min(listvalues)
    if minValue>=3:
        return("null")
    
    wordMatches = []
    i = 0
    for index in listvalues:
        if index==minValue:
           wordMatches.append(wordList[i])
        i = i+1
    
    import random
    return(wordMatches[random.randrange(len(wordMatches))])


print(wordMatch('afgcan', ['area']))