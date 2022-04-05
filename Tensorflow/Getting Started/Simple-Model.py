import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


w = tf.Variable(tf.random.normal((3, 2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = np.array([[1., 2., 3.]])

with tf.GradientTape(persistent=True) as tape:
	y = x @ w + b
	loss = tf.reduce_mean(y**2)
  
[dl_dw, dl_db] = tape.gradient(loss, [w, b])

Variables = {'w': w, 'b': b}
grad = tape.gradient(loss, Variables)

print (grad)
