import torch

x =torch.eye(1)
print(x)
x.requires_grad_(True)

'''
y = x**3
y.backward(x)
'''

def function(x):
    return x**2

loss = function(4*x)# + function(2*x)
loss.backward()
print(x.grad)
