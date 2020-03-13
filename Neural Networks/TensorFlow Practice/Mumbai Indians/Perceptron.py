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
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
from matplotlib import animation, rc
from IPython.display import HTML

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



""" Plotting Data """

plt.plot(X_train.T ,'+')
plt.plot(X_test.T ,'.')
plt.xticks(rotation='vertical')
plt.show()



""" Binarisation """
X_binarised_train = X_train.apply(pd.cut,bins=2,labels=[0,1])
X_binarised_test = X_test.apply(pd.cut,bins=2,labels=[0,1])
X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values		

plt.plot(X_binarised_train.T ,'+')
plt.plot(X_binarised_test.T ,'.')
plt.xticks(rotation='vertical')
plt.show()


""" Perceptron Class	""" 

class Perceptron:
	
	def __init__(self):
		self.w = None
		self.b = None
	
	def model(self,x):
		return 1 if (np.dot(self.w,x) >= self.b) else 0
		
	def predict(self,X):
		Y = []
		for x in X:
			output = self.model(x)
			Y.append(output)
		return np.array(Y)
		
	def fit(self,X,Y,epochs=1,lr=0.01):
		self.w = np.ones(X.shape[1])
		self.b = 0
		accuracy = {}
		max_accuracy = 0
		wt_matrix = []
		
		for i in range(epochs):
			for x,y in zip(X,Y):
				y_pred = self.model(x)
				
				if (y==1 and y_pred==0):
					self.w = self.w + lr*x
					self.b = self.b + lr*1
				elif (y==0 and y_pred==1):
					self.w = self.w - x
					self.b = self.b - 1	
			
			wt_matrix.append(self.w)
			
			accuracy[i] = accuracy_score(self.predict(X), Y)
			if (accuracy[i] > max_accuracy):
				max_accuracy = accuracy[i]
				chkptw = self.w
				chkptb = self.b
				
		self.w = chkptw
		self.b = chkptb
		
		print (max_accuracy)
		plt.plot(list(accuracy.values()))
		plt.show()
		
		return np.asarray(wt_matrix)

			
perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train, Y_train,25,0.11)

plt.plot(perceptron.w)
plt.show()

Y_pred_train = perceptron.predict(X_train)
accuracy_train = accuracy_score(Y_pred_train, Y_train)
print ("Training Accuracy: ", accuracy_train)

Y_pred_test = perceptron.predict(X_test)
accuracy_test = accuracy_score(Y_pred_test, Y_test)
print ("Test Accuracy: ", accuracy_test)



""" Animation of Weights """

fig, ax = plt.subplots()
ax.set_xlim((0, wt_matrix.shape[1]))
ax.set_ylim((-300, 300))
line, = ax.plot([],[],lw=2)

def animate(i):
	a = wt_matrix.shape[0]
	for i in range(a):
		x = np.asarray(list(range(wt_matrix.shape[1])))
		y = wt_matrix[i]
		line.set_data(x,y)
		return (line,)


anim = animation.FuncAnimation(fig, animate, frames=100,interval=200, blit=True)
#anim.save('Weights.gif')
