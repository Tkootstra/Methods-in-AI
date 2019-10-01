# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:53:23 2019

@author: harme
"""

def restaurantLookup (attributeList, attributeNames):
    if len(attributeList)!=len(attributeNames) or len(attributeList)>3:
        return 'null'
    
    filestring = 'C:\\Users\\harme\\Documents\\Studie\\AI\Master\\Methods in AI research\\Code\\restaurantinfo.csv'
    import pandas as pd
    data = pd.read_csv(filestring)
    
    if len(attributeList)==1:
       return (data.loc[data[attributeList[0]] == attributeNames[0]])
    if len(attributeList)==2:
       return (data.loc[(data[attributeList[0]] == attributeNames[0]) & (data[attributeList[1]] == attributeNames[1])])
    if len(attributeList)==3:
       return (data.loc[(data[attributeList[0]] == attributeNames[0]) & (data[attributeList[1]] == attributeNames[1])  & (data[attributeList[2]] == attributeNames[2])])


