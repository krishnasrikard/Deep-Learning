# Disabling all debugging logs using os.environ
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Source: https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# References
'''
https://stackoverflow.com/questions/51932767/how-to-interpret-clearly-the-meaning-of-the-units-parameter-in-keras
https://stackoverflow.com/questions/38714959/understanding-keras-lstms/50235563#50235563
'''

# Notations
'''
SlSL : Single-Layered Single-LSTM
SlML : Single-Layered Multi-LSTM
MlSL : Multi-Layered Single-LSTM
MlML : Multi-Layered Multi-LSTM
'''

Data = np.array([[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]).reshape((1,3,2))
print ('Data:\n',Data)


# Single-Layered Single-LSTM Model-1
print ('--------------------------------Single-Layered Single-LSTM Model-1------------------------------------')
inputs1 = Input(shape=(3,2))
x = LSTM(1)(inputs1)
SlSL_1 = Model(inputs=inputs1, outputs=x, name='SlSL_1')				

print (SlSL_1.summary())
print ('SlSL_1\n',SlSL_1.predict(Data))




# Single-Layered Single-LSTM Model-2 i.e with return_sequences=True
"""
hidden state output for each input time step.
"""
print ('-------------------Single-Layered Single-LSTM Model-2 with return_sequences=True----------------------')
inputs1 = Input(shape=(3,2))
x = LSTM(1,return_sequences=True)(inputs1)
SlSL_2 = Model(inputs=inputs1, outputs=x, name='SlSL_2')				

print (SlSL_2.summary())
print ('SlSL_2\n',SlSL_2.predict(Data))




# Single-Layered Single-LSTM Model-3 i.e with return_states=True
"""
1. The LSTM hidden state output for the last time step.
2. The LSTM hidden state output for the last time step (again).
3. The LSTM cell state for the last time step.
"""
print ('---------------------Single-Layered Single-LSTM Model-3 with return_states=True-----------------------')
inputs1 = Input(shape=(3,2))
x = LSTM(1,return_state=True)(inputs1)
SlSL_3 = Model(inputs=inputs1, outputs=x, name='SlSL_3')				

print (SlSL_3.summary())
print ('SlSL_3\n',SlSL_3.predict(Data))




# Single-Layered Single-LSTM Model-4 i.e with return_states=True and return_sequences=True
"""
The layer returns the hidden state for each input time step, then separately, 
the hidden state output for the last time step and the cell state for the last input time step.
"""
print ('-------Single-Layered Single-LSTM Model-4 with return_states=True and return_sequences=True-----------')
inputs1 = Input(shape=(3,2))
x = LSTM(1,return_state=True,return_sequences=True)(inputs1)
SlSL_4 = Model(inputs=inputs1, outputs=x, name='SlSL_4')				

print (SlSL_4.summary())
print ('SlSL_4\n',SlSL_4.predict(Data))




# Single-Layered Multi-LSTM Model
print ('----------------------------------Single-Layered Multi-LSTM Model-------------------------------------')
inputs1 = Input(shape=(3,2))
x = LSTM(4)(inputs1)
SlML = Model(inputs=inputs1, outputs=x, name='SlML')	
			
print (SlML.summary())
print ('SlML\n',SlML.predict(Data))




# Multi-Layered Single-LSTM Model
print ('----------------------------------Multi-Layered Single-LSTM Model-------------------------------------')
inputs1 = Input(shape=(3,2))
a = LSTM(1, return_sequences=True)(inputs1)
x = LSTM(1)(a)
MlSL = Model(inputs=inputs1, outputs=x, name='MlSL')	
			
print (MlSL.summary())
print ('MlSL\n',MlSL.predict(Data))




# Multi-Layered Multi-LSTM Model
print ('----------------------------------Multi-Layered Multi-LSTM Model-------------------------------------')
inputs1 = Input(shape=(3,2))
a = LSTM(4, return_sequences=True)(inputs1)
x = LSTM(4)(a)
MlML = Model(inputs=inputs1, outputs=x, name='MlML')	
			
print (MlML.summary())
print ('MlML\n',MlML.predict(Data))
