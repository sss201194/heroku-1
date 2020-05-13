# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:51:50 2020

@author: sande
"""

import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv('Salary.csv')
x=dataset.iloc[:,:2]
y=dataset.iloc[:,-1]


from sklearn.linear_model import LinearRegression
regressor=LinearRegression

regressor.fit(x,y)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

