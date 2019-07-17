import torch
in_planes = torch.tensor([[4, 2, -1],[-6, 0, 5],[3, 2, 2]])
out_planes = torch.tensor([[0, 1, 2],[1, -1, 0],[1, 0, -2]])
torch.conv2d(in_planes, out_planes, stride=2, padding=1)
#print(cnn)
