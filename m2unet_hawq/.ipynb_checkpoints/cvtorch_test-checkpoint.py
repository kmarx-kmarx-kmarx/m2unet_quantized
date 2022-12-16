from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable


# net = ptcv_get_model("resnet18", pretrained=True)
# x = Variable(torch.randn(1, 3, 224, 224))
# y = net(x)
# print(y)

# Try with a u net architecture
net2 = ptcv_get_model("unet_cityscapes", pretrained = False)
x = Variable(torch.randn(1,3,512,512))
y = net2(x)
print(y)