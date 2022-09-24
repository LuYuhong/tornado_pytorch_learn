import torch
foo = torch.rand(1, 3, 224, 224).to('mps')

device = torch.device('mps')
foo = foo.to(device)
print('foo')