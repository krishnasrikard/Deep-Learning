import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Building a Large Linear Model
print ("..............................................................")
print ("Building a Large Linear Model")
print ("..............................................................")


# Without GPU
learning_rate = 0.001
N = 10000
epochs = 500

w = torch.zeros([N], requires_grad=True)
b = torch.ones([1], requires_grad=True)

# print(torch.mean(w).item(), b.item())

x = torch.randn([N])
y = torch.dot(3*torch.ones([N]), x) - 2

for i in range(epochs):
	
	y_hat = torch.dot(w, x) + b
	loss = torch.sum((y_hat - y)**2)
  
	loss.backward()
  
	with torch.no_grad():
		w -= learning_rate * w.grad
		b -= learning_rate * b.grad
    
		w.grad.zero_()
		b.grad.zero_()

	print ("Final Values of w and b are",torch.mean(w).item(),"and",b.item(), "respectively")
print ("--------------------------------------------------------------")


# With GPU
cuda0 = torch.device('cuda:0')
learning_rate = 0.001
N = 10000
epochs = 500

w = torch.zeros([N], requires_grad=True, device=cuda0)
b = torch.ones([1], requires_grad=True, device=cuda0)

# print(torch.mean(w).item(), b.item())

x = torch.randn([N], device=cuda0)
y = torch.dot(3*torch.ones([N], device=cuda0), x) - 2

for i in range(epochs):
  
	y_hat = torch.dot(w, x) + b
	loss = torch.sum((y_hat - y)**2)
  
	loss.backward()
  
	with torch.no_grad():
		w -= learning_rate * w.grad
		b -= learning_rate * b.grad
    
		w.grad.zero_()
		b.grad.zero_()
    
print ("Final Values of w and b are",torch.mean(w).item(),"and",b.item(), "respectively")
print ("--------------------------------------------------------------") 
