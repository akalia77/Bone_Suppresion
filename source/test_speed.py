import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import numpy as np 
from unetTest import UNet
# from unet import UNet


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output



#%%

isCuda= False
# isCuda= True

# Equates to one random 28x28 image
# random_data = torch.rand((1, 1, 28, 28))
random_data = torch.rand((1, 1, 28, 28))

my_nn = Net()


timeS = timeit.default_timer()
result = my_nn(random_data)
timeE =timeit.default_timer()
print("run-nn:: %f sec "%(timeE-timeS))


random_unet = torch.rand((1, 1, 256, 256))

unet_m = UNet(in_channels=1,out_channels=1)

unet_m.eval()


if isCuda == True:
    unet_m= unet_m.cuda()
    random_unet= random_unet.cuda()

    print("run-mode: Cuda----")
else:
    unet_m= unet_m.cpu()
    random_unet= random_unet.cpu()

    print("run-mode: cpu----")


# mode TEST 

# d1=unet_m._block1(random_unet) #in 1,256, 256 :: out 48,128,128
# print(d1.size())
# d2=unet_m._block2(d1)  # in 48,128,128 :: out 48, 64,64
# print(d2.size())
# u1=unet_m._block3(d2)              # in 48,64, 64 :: out 48, 128, 128
# print(u1.size())
# c1 = torch.cat((u1, d1), dim=1)# in 48, 128,128 :: out 1,256 256 
# print(c1.size())
# u2=unet_m._block5_1(c1)
# print(u1.size())
# c2=torch.cat((u2, random_unet), dim=1)
# out= unet_m._block6_1(c2)

for cnt in range(3):
    timeS = timeit.default_timer()

    result = unet_m(random_unet).detach()

    timeE =timeit.default_timer()
    print("run-unet:: %f sec "%(timeE-timeS))



if isCuda == True:
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = unet_m(random_unet)
else:
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        result = unet_m(random_unet)


print(prof.key_averages())


print (result)



