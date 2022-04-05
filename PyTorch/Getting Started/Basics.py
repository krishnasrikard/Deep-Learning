import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import time


# Creating Tensors
print ("..............................................................")
print ("Tensors")
print ("..............................................................")

x = tr.tensor([[0,1],[2,3],[4,5]])
print (x)
print ("--------------------------------------------------------------")

z = tr.ones(3,2)
y = tr.zeros(3,2)
x = tr.rand(3,2)
print (x+y+z)
print ("--------------------------------------------------------------")

z = tr.empty(3,2)
print (z)
y = tr.zeros_like(z)
print (y)
print ("--------------------------------------------------------------")

a = tr.linspace(0,1,5)
print (a)
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
print (x.size())
print (x[:,1])
print (x[0,:])
print (x[1,1])
print (x[1,1].item())
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
print (x)
print (x.view(2,3))
print (x.view(6,-1))
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
y = tr.tensor([[00,10],[20,30],[40,50]])
print (x)
print (y)
print (x+y)
print (x-y)
print (x*y)
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
y = tr.tensor([[00,10],[20,30],[40,50]])
z = y.add(x)
print (z)
print (y)
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
y = tr.tensor([[00,10],[20,30],[40,50]])
z = y.add_(x)
print (z)
print (y)
print ("--------------------------------------------------------------")

x = tr.tensor([[0,1],[2,3],[4,5]])
y = tr.tensor([[00,10],[20,30],[40,50]])
x = x+1
print (x)
print ("--------------------------------------------------------------")
print()


# Conversion of NumPy and PyTorch
print ("..............................................................")
print ("Conversion of NumPy and PyTorch")
print ("..............................................................")

x_tr = tr.tensor([[0,1],[2,3],[4,5]])
y_np = np.array([[00,10],[20,30],[40,50]])

x_np = x_tr.numpy()
print (type(x_tr)," ----> ",type(x_np))
print (x_np)
print ()

y_tr = tr.from_numpy(y_np)
print (type(y_np)," ----> ",type(y_tr))
print (y_tr)
print ("--------------------------------------------------------------")

np.add(y_np,1, out=y_np)
print (y_np)
print (y_tr)
print ("--------------------------------------------------------------")
print()


# Time of Execution between NumPy and PyTorch without GPU
print ("..............................................................")
print ("Time of Execution between NumPy and PyTorch without GPU")
print ("..............................................................")

start = time.time()
for i in range(10):
	x = np.random.rand(100,100)
	y = np.random.rand(100,100)
	z = x*y
end = time.time()
print (end - start)

start = time.time()
for i in range(10):
	x = tr.randn(100,100)
	y = tr.randn(100,100)
	z = x*y
end = time.time()
print (end - start)

"""
start = time.time()
for i in range(10):
	x = np.random.rand(10000,10000)
	y = np.random.rand(10000,10000)
	z = x+y
end = time.time()
print (end - start)

start = time.time()
for i in range(10):
	x = tr.randn(10000,10000)
	y = tr.randn(10000,10000)
	z = x+y
end = time.time()
print (end - start)
"""
print ("..............................................................")
print ()


# Time of Execution between NumPy and PyTorch with GPU
print ("..............................................................")
print ("Time of Execution between NumPy and PyTorch with GPU")
print ("..............................................................")
print ("No.of CUDA devices: ",tr.cuda.device_count())
print (tr.cuda.device(0))
print ("Name of Device is ",tr.cuda.get_device_name(0))

cuda0 = tr.device('cuda:0')

start = time.time()
for i in range(10):
	x = tr.randn([100,100],device=cuda0)
	y = tr.randn([100,100],device=cuda0)
	z = x*y
end = time.time()
print (end - start)

"""
start = time.time()
for i in range(10):
	x = np.random.rand(10000,10000)
	y = np.random.rand(10000,10000)
	z = x+y
end = time.time()
print (end - start)

start = time.time()
for i in range(10):
	x = tr.randn(10000,10000)
	y = tr.randn(10000,10000)
	z = x+y
end = time.time()
print (end - start)
"""
print ("..............................................................")
