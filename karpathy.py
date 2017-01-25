import math
from random import random

x, y = -2, 3


# Real Valued circuits 

def forwardMultiplyGate(a, b):
	return a * b

def forwardAddGate(a, b):
	return a + b

print(forwardMultiplyGate(x, y))

# print(forwardMultiplyGate(-2, 3))

#Tweak input slightly to increase the output

# 1. Random Local Search
def randomLocalSearch(x, y):
	tweak_amount = 0.01
	best_out = -math.inf 
	for k in range(100):
		x_try = x + tweak_amount * (random() * 2 - 1)
		y_try = y + tweak_amount * (random() * 2 - 1)
		out = forwardMultiplyGate(x_try, y_try)
		if out > best_out:
			best_out = out 
			best_x, best_y = x_try, y_try
	print(best_out)
	print(best_x, best_y)
	return best_out 

randomLocalSearch(x, y)


# 2. Numerical Gradient 
def numericalGradient(x, y):
	h = 0.0001
	out = forwardMultiplyGate(x, y)

	#Derivative WRT x
	xph = x + h
	out2 = forwardMultiplyGate(xph, y)
	x_derivative = (out2 - out) / h

	#Derivative WRT y
	yph = y + h
	out3 = forwardMultiplyGate(x, yph)
	y_derivative = (out3 - out) / h

	step_size = 0.01
	x += step_size * x_derivative
	y += step_size * y_derivative

	out_new = forwardMultiplyGate(x, y)
	return out_new 

print(numericalGradient(x, y))

# 3. Analytical Gradient
def analyticalGradient(x, y):
	step_size = 0.01
	x_gradient = y
	y_gradient = x 

	x += step_size * x_gradient
	y += step_size * y_gradient
	out_new = forwardMultiplyGate(x, y)
	return out_new 

print(analyticalGradient(x, y))

'''
In practice, all NN libraries compute the analytical gradient but the correctness of the
implementation is verified by comparing it to the numerical gradient

Evaluating the gradient during backprop or during backward pass will cost almost as much
as evaluating forward pass
'''

#Recursive case - Circuits with multiple gates
'''
Multiple gates in conjunction with one another in a circuit
Refer diagram from article
'''

def forwardCircuit(x, y, z):
	q = forwardAddGate(x, y)
	f = forwardMultiplyGate(q, z)
	return q, f 

x, y, z = -2, 5, -4
_, f = forwardCircuit(x, y, z)
print(f)

#Backpropagation

def backpropagation(x, y, z):
	q, f = forwardCircuit(x, y, z)
	print(q, f)
	#Multiplication gate
	derivative_f_wrt_z = q 
	derivative_f_wrt_q = z 

	#Add gate 
	derivative_q_wrt_x = 1.0 
	derivative_q_wrt_y = 1.0 

	#F with respect to x and y
	derivative_f_wrt_x = derivative_q_wrt_y * derivative_f_wrt_q 
	derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q 

	gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]
	print(gradient_f_wrt_xyz)
	#return gradient_f_wrt_xyz

	#Inputs respond to the force/tug
	x += step_size * derivative_f_wrt_x
	y += step_size * derivative_f_wrt_y
	z += step_size * derivative_f_wrt_z

	q, f = forwardCircuit(x, y, z)
	print(q, f)

def numericalGradient(x, y, z):
	h = 0.0001
	x_derivative = (forward_circuit(x+h,y,z) - forward_circuit(x,y,z)) / h
	y_derivative = (forward_circuit(x,y+h,z) - forward_circuit(x,y,z)) / h
	z_derivative = (forward_circuit(x,y,z+h) - forward_circuit(x,y,z)) / h
	return [x_derivative, y_derivative, z_derivative]


#Single Neuron
'''
2 dimensional neuron that computes following function
f(x,y,a,b,c) = sigma(ax+by+c)
sigma - sigmoid function
Squashing function - very negative values are squashed towards zero and very positive
values are squashed towards 1

sigma(x) = 1 / 1 + e^-x

gradient wrt single input
d(sigma(x))/dx = sigma(x)/(1-sigma(x))
'''
class Unit(object):
	def __init__(self, value=0, grad=0):
		self.value = value 
		self.grad = grad 

#Multiply gate class
class multiplyGate(object):
	def __init__(self):
		self.u0 = Unit()
		self.u1 = Unit()
		self.utop = None

	def forward(self, u0, u1):
		self.u0 = u0
		self.u1 = u1
		self.utop = Unit(u0.value * u1.value, 0.0)
		return self.utop

	def backward(self):
		self.u0.grad += self.u1.value * self.utop.grad 
		self.u1.grad += self.u0.value * self.utop.grad 

#Addition Gate Class
class addGate(object):
	def __init__(self):

	def forward(self, u0, u1):
		self.u0 = u0
		self.u1 = u1 
		self.utop = Unit(u0.value + u1.value, 0.0)

	def backward(self):
		#Add gate. Derivative wrt both inputs is 1
		self.u0.grad += 1 * self.utop.grad 
		self.u1.grad += 1 * self.utop.grad 

#Sigmoid Gate class
class sigmoidGate(object):
	def __init__(self):
		self.sigmoid = None

	def sig(self, x):
		self.sigmoid = 1 / (1 + math.exp(-x))
		return self.sigmoid 

	def forward(self, u0):
		self.u0 = u0 
		self.utop = Unit()

