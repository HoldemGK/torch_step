import torch
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize'] = (13.0 , 5.0)

x_train = torch.rand(100)
x_train = x_train * 20.0 -10.0
y_train = torch.sin(x_train)
plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('$y = sin(x)$')

noise = torch.randn(y_train.shape) / 5.
plt.plot(x_train.numpy(), noise.numpy(), 'o')
plt.axis([-10, 10, -1, 1])
plt.title('Gaussian noise')

y_train = y_train + noise
plt.plot(x_train.numpy(), y_train.numpy(), 'o')
plt.title('noisy sin(x)')
plt.xlabel('x_train')
plt.ylabel('y_train')

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

tmp = torch.Tensor([1,2,3])
print(tmp)
print(tmp.unsqueeze(1))

# Validation dataset
x_validation = torch.linspace(-10, 10, 100)
y_validation = torch.sin(x_validation.data)
plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
plt.title('sin(x)')
plt.xlabel('x_validation')
plt.ylabel('y_validation')

x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

#Model construction
