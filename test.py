import torch.nn as nn

vob = 5,
dim = 2
a = nn.Linear(2,5,bias = False)
print(a.weight.shape)

b = nn.Embedding(5,2)
print(b.weight.shape)