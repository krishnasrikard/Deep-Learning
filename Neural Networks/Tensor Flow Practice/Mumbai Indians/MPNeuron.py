from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
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



""" Analysis Data """

print(df["Result"].value_counts())
#	print (df.target_names)
print ("--------------------------------------------------------------------------")
print (df.groupby("Result").mean())
print ("--------------------------------------------------------------------------")



""" Splitting Data for Training and Validation. """

X = df.drop('Result',axis=1)
X = df.drop('Total',axis=1)
Y = df["Result"]


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=1)
# Inorder to split cases equally among Train and Test sets and not to generate data always Randomly


print (X.shape, X_train.shape, X_test.shape)
print ("--------------------------------------------------------------------------")

print (Y.mean(), Y_train.mean(), Y_test.mean())
print ("--------------------------------------------------------------------------")

print (X.mean(), X_train.mean(), X_test.mean())
print ("--------------------------------------------------------------------------")



""" Plotting Data """

plt.plot(X_train.T ,'+')
plt.plot(X_test.T ,'.')
plt.xticks(rotation='vertical')
plt.show()



""" Binarisation """
X_binarised_train = X_train.apply(pd.cut,bins=2,labels=[0,1])
X_binarised_test = X_test.apply(pd.cut,bins=2,labels=[0,1])
X_binarised_train = X_binarised_train.values
X_binarised_test = X_binarised_test.values

plt.plot(X_binarised_train.T ,'+')
plt.plot(X_binarised_test.T ,'.')
plt.xticks(rotation='vertical')
plt.show()


"""  Model """

acc = []
for b in range(X_binarised_train.shape[1] + 1):
	Y_pred_train = []
	accurate_rows = 0
	
	for x,y in zip(X_binarised_train, Y_train):
		y_pred = (np.sum(x) >= b)
		Y_pred_train.append(y_pred)
		accurate_rows += (y == y_pred)
	
	acc.append(accurate_rows)
	
b = np.argmax(acc)

Y_pred_test = []
accurate_rows = 0

for x in X_binarised_test:
	y_pred = (np.sum(x) >= b)
	Y_pred_test.append(y_pred)
	
acc.append(accurate_rows)

accuracy_test = accuracy_score(Y_pred_test,Y_test)
print (accuracy_test)
print ("--------------------------------------------------------------------------")


""" MPNeuron Class	""" 

class MPNeuron:
	
	def __init__(self):
		self.b = None
	
	def model(self,x):
		return (sum(x) >= self.b)
		
	def predict(self,X):
		Y = []
		for x in X:
			output = self.model(x)
			Y.append(output)
		return np.array(Y)
		
	def fit(self,X,Y):
		accuracy = {}
		for b in range(X.shape[1] + 1):
			self.b = b
			Y_pred = self.predict(X)
			accuracy[b] = accuracy_score(Y_pred,Y)
		
		best_b = max(accuracy, key=accuracy.get)
		self.b = b
		
		print ("Optimal Value of b is ",best_b)
		print ("Highest Accuracy for b is ",accuracy[best_b])
		
		
mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train,Y_train)

Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred,Y_test)
print ("Test Accuracy is equal to ",accuracy_test)
