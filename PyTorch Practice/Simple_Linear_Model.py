import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import time

# Building a Simple Linear Model
print ("..............................................................")
print ("Building a Simple Linear Model")
print ("..............................................................")

w = tr.tensor([1.0], requires_grad=True)
b = tr.tensor([1.0], requires_grad=True)
lr = 0.03

for i in range(16):
	
	x = tr.randn([20,1], requires_grad=True)
	y = 3*x - 2

	y_hat = w*x + b

	loss = tr.sum((y_hat-y)**2)
	loss.backward()
	
	with tr.no_grad():									# Inorder to avoid Cross Computation
		w -= lr*w.grad
		b -= lr*b.grad
		
		w.grad.zero_()									# Setting w.grad and b.grad to zeros for next Iteration
		b.grad.zero_()
		
	print (w.item(),b.item())
	
print ("Final Values of w and b are",w.item(),"and",b.item(), "respectively")
print ("--------------------------------------------------------------")

