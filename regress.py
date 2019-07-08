import torch

class RegressionNet(torch.nn.Module):
    def __init__(self, n_hiden_neurons):
        super(RegressionNet, self).__init__()
        NUMNE = n_hiden_neurons
        self.fc1 = torch.nn.Linear(1, NUMNE)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(NUMNE, NUMNE)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(NUMNE, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

net = RegressionNet(20)

def target_function(x):
    return 2**x * torch.sin(2**-x)

# ------Dataset preparation start--------:
x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:

def metric(pred, target):
    return (pred - target).abs().mean()

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def loss(pred, target):
    return (pred - target).abs().mean()
