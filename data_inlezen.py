# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:40:54 2019

@author: timok
"""

import pandas as pd
import json
rest_info = pd.read_csv('restaurantinfo (1).csv')

json_file = 'ontology_dstc2.json'

with open(json_file) as file:
    data = json.load(file)
    
for colname in ['pricerange', 'area', 'food']:
    print(rest_info[colname].unique())