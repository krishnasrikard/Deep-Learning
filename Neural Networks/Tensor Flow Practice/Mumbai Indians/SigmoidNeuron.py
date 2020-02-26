"""
The following code utilises TesnsorFlow 1.x
"""
from __future__ import print_function

import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from matplotlib import animation, rc
from IPython.display import HTML
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#tf.logging.set_verbosity(tf.logging.ERROR)
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


""" Splitting Data for Training and Validation. """

X = df.drop('Result',axis=1)
Y = df["Result"]
print (Y.value_counts())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=1)
# Inorder to split cases equally among Train and Test sets and not to generate data always Randomly

Y_train = Y_train.values
Y_test = Y_test.values

print (X.shape, X_train.shape, X_test.shape)
print ("--------------------------------------------------------------------------")

print (Y.mean(), Y_train.mean(), Y_test.mean())
print ("--------------------------------------------------------------------------")

print (X.mean(), X_train.mean(), X_test.mean())
print ("--------------------------------------------------------------------------")

""" Normalisation fo Data """
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

minmaxscalar = MinMaxScaler()
Y_train = minmaxscalar.fit_transform(Y_train.reshape(-1,1))
Y_test = minmaxscalar.transform(Y_test.reshape(-1,1))

""" -----------------------------------------------------------------------------------------------"""

def sigmoid(x,w,b):
	return 1.0/(1.0 + np.exp(-(np.dot(w,x) + b)))

def calculate_loss(X,Y,w,b):
	loss = 0
	for x,y in zip(X,Y):
		loss += (y - sigmoid(x,w,b))**2
		
	return loss
	
def Plot_Sigmoid(X,Y,S):
	X1 = np.linspace(-10,10,100)
	X2 = np.linspace(-10,10,100)
	
	Xm1,Xm2 = np.meshgrid(X1,X2)
	Ym = np.zeros(np.shape(Xm1))
	
	for i in range(X2.size):
		for j in range(X1.size):
			val = np.asarray([X1[j],X2[i]])
			Ym[i,j] = S.sigmoid(S.perceptron(val))
		plt.contourf(Xm1,Xm2,Ym)
		plt.scatter(X[:,0],X[:,1])
		plt.plot()
		
class SigmoidNeuron:
	
	def __init__(self):
		self.w = None
		self.b = None
		
	def perceptron(self, x):
		return np.dot(self.w,x.T) + self.b
	
	def sigmoid(self, x):
		return 1.0/(1.0 + np.exp(-x))
		
	def grad_w(self, x, y):
		y_pred = self.sigmoid(self.perceptron(x))
		return (y_pred-y)*(y_pred)*(1-y_pred)*x
		
	def grad_b(self, x, y):
		y_pred = self.sigmoid(self.perceptron(x))
		return (y_pred-y)*(y_pred)*(1-y_pred)
	
	def fit(self,X,Y,epochs=1,learning_rate=1,intialise=True,display_loss=False):
		
		if intialise:
			self.w = np.random.randn(1,X.shape[1])
			self.b = 0
		
		if display_loss:
			loss={}
			
		for i in range(epochs):
			dw = 0
			db = 0
			for x,y in zip(X,Y):
				dw += self.grad_w(x,y)
				db += self.grad_b(x,y)
			
			self.w -= learning_rate * dw
			self.b -= learning_rate * db
			
			if display_loss:
				Y_pred = self.sigmoid(self.perceptron(X))
				loss[i] = mean_squared_error(Y_pred.T,Y)
				
		if display_loss:
			plt.plot(list(loss.values()))
			plt.xlabel("Epochs")
			plt.ylabel("Mean Squared Error")
			plt.show()		
				
			
	def predict(self,X):
		Y_pred = []
		for x in X:
			y_pred = self.sigmoid(self.perceptron(x))
			Y_pred.append(y_pred)
		return np.asarray(Y_pred)
		
			
X = np.zeros((X_train.shape[0],2))
X[:,0] = X_train[:,0]
X[:,1] = X_train[:,5]
Y = np.asarray(Y_train)

X_testdata = np.zeros((X_test.shape[0],2))
X_testdata[:,0] = X_test[:,0]
X_testdata[:,1] = X_test[:,5]
Y_test = Y_test

S = SigmoidNeuron()
S.fit(X,Y,100,0.06,display_loss=True)

Y_pred_train = S.predict(X)
Y_pred_test = S.predict(X_testdata)

threshold = 0.5

Y_pred_train_binarised = (Y_pred_train > threshold).astype("int").ravel()
Y_pred_test_binarised = (Y_pred_test > threshold).astype("int").ravel()

accuracy_train = accuracy_score(Y,Y_pred_train_binarised)
accuracy_test = accuracy_score(Y_test,Y_pred_test_binarised)

print (accuracy_train,accuracy_test)

print (S.w,S.b)
Plot_Sigmoid(X,Y,S)
plt.show()
