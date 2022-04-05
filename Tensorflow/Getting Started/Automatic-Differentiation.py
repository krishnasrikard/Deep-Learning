import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

"""
Automatic Differentation
"""
print ("-"*20 + "Automatic Differentation" + "-"*20)

x = tf.Variable(10.0)
with tf.GradientTape() as tape:
	y = x**3 - x**2 + x - 1
	
# dy = 3x**2 - 2x + 1
dy_dx = tape.gradient(y, x)
print (dy_dx.numpy())


"""
Understanding types of Variables
"""
print ("-"*20 + "Understanding types of Variables" + "-"*20)

# A trainable variable
x0 = tf.Variable(3.0, name='x0')
# Not trainable
x1 = tf.Variable(3.0, name='x1', trainable=False)
# Not a Variable: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.0
# Not a variable
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
	y = (x0**2) + (x1**2) + (x2**2) + x3

grad = tape.gradient(y, [x0, x1, x2, x3])

for g in grad:
	print(g)


"""
Gradient Watch
"""
print ("-"*20 + "Gradient Watch" + "-"*20)

# A trainable variable
x0 = tf.Variable(3.0, name='x0')
# Not a variable
x1 = tf.constant(6.0, name='x1')

with tf.GradientTape() as tape:
	tape.watch(x1)
	y = (x0**2) + (x1**2)

grad = tape.gradient(y, [x0, x1])

for g in grad:
	print(g)
