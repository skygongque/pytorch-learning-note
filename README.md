# pytorch学习笔记

## CROSSENTROPYLOSS

```
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
```

## torch.argmax
> torch.argmax(input, dim, keepdim=False) → LongTensor

Returns the indices of the maximum values of a tensor across a dimension.
```
import torch
a = torch.randn(3, 4)
# dim=0 每列最大值 dim=1 每行最大值
b = torch.argmax(a, dim=0)
print(a)
print(b)
```
## view 改变tensor shape

example
```
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])

>>> a = torch.randn(1, 2, 3, 4)
>>> a.size()
torch.Size([1, 2, 3, 4])
>>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
>>> b.size()
torch.Size([1, 3, 2, 4])
>>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
>>> c.size()
torch.Size([1, 3, 2, 4])
>>> torch.equal(b, c)
False

x = torch.randn(4, 4)
x.view(-1) #降到1维

```

## one_hot_encode

第一种方法  

```
import torch
import torch.nn as nn
import string


characters = string.digits

target_str = '1234'
tensor_a = torch.tensor([characters.find(x) for x in target_str])
N = tensor_a.size(0)
num_class = len(characters)
one_hot = torch.zeros(N,num_class)
one_hot.scatter_(dim=1,index=torch.unsqueeze(tensor_a,dim=1),src=torch.ones(N,num_class))
print(one_hot)
```

第二种方法

```
target_str = '1234'
target = []
for char in target_str:
    vec = [0.0] * len(characters)
    vec[characters.find(char)] = 1.0
    target += vec

tensor_b = torch.tensor(target)
tensor_b = tensor_b.view(-1,len(characters))
print(tensor_b)
```
