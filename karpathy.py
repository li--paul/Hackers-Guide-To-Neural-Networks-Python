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
q, f = forwardCircuit(x, y, z)
derivative_f_wrt_z = q 
derivative_f_wrt_q = z 

derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0 

derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q 
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

#Let inputs respond to the force/tug 
step_size = 0.01 

def backpropagation():
	q, f = forwardCircuit(x, y, z)
	#Multiplication gate
	derivative_f_wrt_z = q 
	derivative_f_wrt_q = z 

	#Add gate 
	derivative_q_wrt_x = 1.0 
	derivative_q_wrt_y = 1.0 

	#F with respect to x and y
	derivative_f_wrt_x = derivative_q_wrt_y * derivative_f_wrt_q 
	derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q 

	gradient_f_wrt_xyz = [gradient_f_wrt_x, gradient_f_wrt_y, derivative_f_wrt_z]


