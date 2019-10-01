# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:36:43 2019

@author: Tim.O
"""
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

filename = 'finalized_model.sav'
baselineModel = pickle.load(open(filename, 'rb'))

def define_word_numbers(wordList):
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

full_data = pd.read_csv('full_data_dialogact_content.csv')
full_data['utterance_content'] = full_data['utterance_content'].apply(lambda words: words.lower())

all_words = []

for sentence in list(full_data['utterance_content']):
    for word in sentence.split():
        all_words.append(word.lower())
        

    
words, numbers = define_word_numbers(all_words)

all_features = np.empty([len(full_data), len(words)])

all_sentences = list(full_data['utterance_content'])
for i in range(len(all_sentences)):
    sentence = all_sentences[i]
    vector = vectorizeSentence(sentence.split())
    all_features[i,:] = vector

all_features = pd.DataFrame(all_features)
all_features['label'] = full_data['dialog_act']
all_features = all_features.dropna()

X = all_features.drop('label', axis=1)
y = all_features['label']
#y = LabelBinarizer().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=101)

modelsToTest = [baselineModel, LogisticRegression(), SVC(kernel='linear'), \
                DecisionTreeClassifier(), RandomForestClassifier(n_estimators=200),\
                KNeighborsClassifier(n_neighbors=10)]

modelNames = ['Neural network', 'Logistic regression', 'Support vector machine',\
              'Decision Tree', 'Random Forest', 'KNN']

performances = []
for model,name in zip(modelsToTest, modelNames):
    print('fitting model {}'.format(name))
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))






