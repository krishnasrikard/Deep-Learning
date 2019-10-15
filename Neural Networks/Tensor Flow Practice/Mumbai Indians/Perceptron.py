from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


df = pd.read_csv('MI.csv',sep=',')															#Importing the .csv for data analysis
df = df.reindex(np.random.permutation(df.index))											#Shuffle																	
print (df)
print (type(df)) 
print ("--------------------------------------------------------------------------")
print (df.describe())																		#Gives statitics of the data
print ("--------------------------------------------------------------------------")
df['Opening'] = df['Rohit'] + df['Quinton']
df['Key_Runs'] = df['Rohit'] + df['Quinton'] + df['Surya'] + df['Pollard'] + df['Krunal'] + df['Hardik']		#Creating a column
print (df)
print ("--------------------------------------------------------------------------")



""" Analysis Data """

print(df["Result"].value_counts())
#	print (df.target_names)
print ("--------------------------------------------------------------------------")
print (df.groupby("Result").mean())
print ("--------------------------------------------------------------------------")



""" Splitting Data for Training and Validation. """

X = df.drop('Result',axis=1)
Y = df["Result"]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=1)
# Inorder to split cases equally among Train and Test sets and not to generate data always Randomly


print (X.shape, X_train.shape, X_test.shape)
print ("--------------------------------------------------------------------------")

print (Y.mean(), Y_train.mean(), Y_test.mean())
print ("--------------------------------------------------------------------------")

print (X.mean(), X_train.mean(), X_test.mean())
print ("--------------------------------------------------------------------------")
