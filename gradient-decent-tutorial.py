import torch

# Create a tensor with requires_grad=True so PyTorch tracks operations on it
x = torch.tensor([3.0], dtype=torch.float32, requires_grad=True)

# Define a simple function y = x^2 + 4
y = x**2 + 4

# Math derivative:
# y = x^2 + 4
# dy/dx = 2x
# At x = 3, dy/dx should be 2*3 = 6

# Compute the gradient of y w.r.t x using backward()
y.backward()  ##this is how we compute a gradients  this will help further in optimization phase in model traning 

# PyTorch stores the computed gradient in x.grad
print("Gradient X:", x.grad)



import torch

# Create two tensors x and y
x = torch.tensor(3.0)
y = torch.tensor(7.0)

# Enable gradient tracking for both tensors
x.requires_grad_(True)
y.requires_grad_(True)

# Define z = x^y  (x raised to the power y)
z = x ** y

# Compute gradients of z w.r.t both x and y
# PyTorch will calculate:
# ∂z/∂x = y * x^(y-1)
# ∂z/∂y = x^y * ln(x)
#
# For x = 3 and y = 7:
# z = 3^7 = 2187
# ∂z/∂x = 7 * 3^6 = 7 * 729 = 5103
# ∂z/∂y = 3^7 * ln(3) = 2187 * 1.0986 = ~2401.79

z.backward()

# PyTorch stores gradients in .grad
print("Gradient wrt x:", x.grad)   # Expected ≈ 5103
print("Gradient wrt y:", y.grad)   # Expected ≈ 2401.79


import torch

# x is just input data, so it does NOT need gradients
x = torch.tensor(5.0)

# m and b are parameters we want to learn, so gradients are required
m = torch.tensor(2.0, requires_grad=True)   # slope
b = torch.tensor(3.0, requires_grad=True)   # intercept

# Define the linear function y = m*x + b
y = m * x + b

# Compute gradients of y w.r.t m and b
# Math:
# y = m*x + b
# ∂y/∂m = x
# ∂y/∂b = 1
#
# For x = 5:
# ∂y/∂m = 5
# ∂y/∂b = 1

y.backward()

# PyTorch stores computed gradients in .grad attributes
print("Gradient wrt m:", m.grad)   # Expected = 5
print("Gradient wrt b:", b.grad)   # Expected = 1
