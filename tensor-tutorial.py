import torch as t
import numpy as np

tensor_1 = t.tensor([1,2,3])

print(tensor_1)

#now we will convert a numpy array into a tensor

arr=np.array([1,2,3])

tensor_from_numpy=t.tensor(arr)

print("Tensor created from the Numpy array",tensor_from_numpy)

#now we will create a empty tensor

empty=t.empty(3)
print(empty)

#now we will create a tensor with all the zeros 

zero=t.zeros(2,3)
print("The tensor with zeros",zero)

#now we will create a tensor with all the ones

one=t.ones(2,3)
print("The tensor with all the ones",one)

#now we will create a tensor with all the random values

random=t.rand(3,3)
print("Random tensor : ",random[1,1])

#now we will create a tensor with all the values between a range

range_tensor=t.arange(start=0,end=100,step=10)

print(range_tensor)

#now we will learn how to reshape a tensor

reshape_tensor=t.rand(5,4)
print(reshape_tensor)

#now we will use the view

# new_shape_tensor=reshape_tensor.view(-1,2,3)
# print("New Reshape tensor",new_shape_tensor)

#now we will perform atrithmatic operation on the tensors

a=t.tensor([1,2,3])
b=t.tensor([4,5,6])

result=t.add(b,a)
print(result)

#now we are going to multiply the tensors with linear algebra rules and regulation

mul_a=t.Tensor([[1,2],[4,5]])
mul_b=t.Tensor([[1,2],[4,5]])

result=t.mm(mul_a,mul_b)
print(result)

#now we will look into dtypes

print(mul_a.dtype)


#now we will design a device agnostic code using Pytorch

print(mul_a.device)

device=t.device('cuda' if t.cuda.is_available() else 'cpu')

