import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import time

x = tr.ones([3,2],requires_grad=True)
print ("x = ",x)

y = x+5
print ("y = ",y)

z = y*y + 3
print ("z = ",z)

t = tr.sum(z)
print ("t = ",t)

t.backward()
print (x.grad)
print ()


x = tr.ones([3, 2], requires_grad=True)
y = x + 5
r = 1/(1 + tr.exp(-y))
print(r)
s = tr.sum(r)
s.backward()
print(x.grad)


x = tr.ones([3, 2], requires_grad=True)
y = x + 5
r = 1/(1 + tr.exp(-y))
a = tr.ones([3, 2])
r.backward(a)
print(x.grad)
