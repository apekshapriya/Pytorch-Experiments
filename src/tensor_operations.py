import torch

# Initializing tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32,
                         device=device, requires_grad = True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initializing methods
x = torch.empty(size = (3,3))   # uninitialized data, random
print(x)
x = torch.zeros((3,3))
print(x)

x = torch.ones((3,3))
x = torch.eye(5,5) # I
x = torch.rand(3,3)
# print(x)

# Advance initialization methods
x = torch.arange(start=0, end=5, step=2)   #0 inclusive
x = torch.linspace(start=0.1, end = 1, steps=5)  # 0.1 and 1 inclusive
# print(x)
x = torch.empty(size = (1,5)).normal_(mean=0, std=1)
x = torch.diag(torch.ones(3))

# how to initialize types
tensor = torch.arange(4)
print(tensor)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())

# array and numpy
import numpy as np
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
print(tensor)
np_array = tensor.numpy()

 
# Tensor Math op

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

z1 = x + y

z = torch.true_divide(x, y)  # element wise div

#inplace op

t = torch.zeros(3)
t.add_(x)   # inplace
t += x  # inplace

t = t+ x # not inplace


# Exponentiation
z = x.pow(2)
x = x ** 2
print(x)

# simple comparison
z = (x>0) & (x<2)   # cant use 'and' here
print(z)

# Matrix

x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
print(x1)
print((x1>0.5) & (x1<0.8))
x1 = torch.empty(size = (2,3, 3))
x1[0][0] = torch.tensor([1,1,1])
print((x1[:,:,0]==1)  &  (x1[:,:,1]==1))


#element wise mult
z = x * y

# dot
z = torch.dot(x, y)


# batch
batch = 32

n =10
m=20
p=30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # (batch, n ,p)


# Example of broadcasting

x1 = torch.rand((5, 5))
x2 = torch.rand((1,5))

print(x1)
# print(x2)

# print(x1-x2)  # broadcasting

# other useful torch operations

sum_x = torch.sum(x1, dim=1)
# print(x)
# print(sum_x)

values, indices = torch.max(x, dim=0)
z = torch.argmax(x, dim=0)

mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x, y)

x1 = torch.sort(x1, dim=0, descending=False)
print(x1)

# indexing tensor


batch_size =  10
features = 25

x = torch.rand((batch_size, features))

print(x[0].shape) , # x[0, :]
print(x[:,0])


# x = torch.arange(start = 0, end = 10, step=1)
x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x, rows, cols)
print(x[rows, cols])

x = torch.arange(10)
print(x[(x<2) & (x>8)])
print(x[x.remainder(2) == 0])

#useful operations

print(torch.where(x>5, x, x *2))
print(torch.tensor([0,0,1,2,2,3,4]).unique())

print(x.ndimension())
print(x.numel())

# Reshaping
x = torch.arange(9)

x_3x3 = x.view(3, 3) # story contigously in memory
x_3x3 = x.reshape(3, 3)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1, x2), dim=0).shape)

z = x1.view(-1) # flatten
batch = 64

x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0,2,1)

x = torch.arange(10)
print(x.unsqueeze(0))
