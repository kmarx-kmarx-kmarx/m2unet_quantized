import torch
import torch.nn as nn
import math


'''
The purpose of this file is to benchmark the speeds of different models before and after quantization
'''
import time

'''
Original m2unet
'''
import torch
import torch.nn as nn
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            # depthwise separable convolution block
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            # Bottleneck with expansion layer
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        
class Encoder(nn.Module):
    """
    14 layers of MobileNetv2 as encoder part
    """
    def __init__(self):
        super(Encoder, self).__init__()
        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
        ]
        # Encoder Part
        input_channel = 32 # number of input channels to first inverted (residual) block
        self.layers = [conv_bn(3, 32, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.layers.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.layers.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # make it nn.Sequential
        self.layers = nn.Sequential(*self.layers)
                
class DecoderBlock(nn.Module):
    """
    Decoder block: upsample and concatenate with features maps from the encoder part
    """
    def __init__(self,up_in_c,x_in_c,upsamplemode='bilinear',expand_ratio=0.15):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False) # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(up_in_c+x_in_c,(x_in_c + up_in_c) // 2,stride=1,expand_ratio=expand_ratio)

    def forward(self,up_in,x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in] , dim=1)
        x = self.ir1(cat_x)
        return x
    
class LastDecoderBlock(nn.Module):
    def __init__(self,x_in_c,upsamplemode='bilinear',expand_ratio=0.15, output_channels=1, activation='linear'):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode=upsamplemode,align_corners=False) # H, W -> 2H, 2W
        self.ir1 = InvertedResidual(x_in_c,16,stride=1,expand_ratio=expand_ratio)
        layers =  [
            nn.Conv2d(16, output_channels, 1, 1, 0, bias=True),
        ]
        if activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif activation == 'softmax':
            layers.append(nn.Softmax(dim=1))
        elif activation == 'linear' or activation is None:
            pass
        else:
            raise NotImplementedError('Activation {} not implemented'.format(activation))
        self.conv = nn.Sequential(
           *layers
        )

    def forward(self,up_in,x_in):
        up_out = self.upsample(up_in)
        cat_x = torch.cat([up_out, x_in] , dim=1)
        x = self.ir1(cat_x)
        x = self.conv(x)
        return x
    
class M2UNet(nn.Module):
        def __init__(self,encoder,upsamplemode='bilinear',output_channels=1, activation="linear", expand_ratio=0.15):
            super(M2UNet,self).__init__()
            encoder = list(encoder.children())[0]
            # Encoder
            self.conv1 = encoder[0:2]
            self.conv2 = encoder[2:4]
            self.conv3 = encoder[4:7]
            self.conv4 = encoder[7:14]
            # Decoder
            self.decode4 = DecoderBlock(96,32,upsamplemode,expand_ratio)
            self.decode3 = DecoderBlock(64,24,upsamplemode,expand_ratio)
            self.decode2 = DecoderBlock(44,16,upsamplemode,expand_ratio)
            self.decode1 = LastDecoderBlock(33,upsamplemode,expand_ratio, output_channels=output_channels, activation=activation)
            # initilaize weights 
            self._init_params()

        def _init_params(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        
        
        def forward(self,x):
            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            decode4 = self.decode4(conv4,conv3)
            decode3 = self.decode3(decode4,conv2)
            decode2 = self.decode2(decode3,conv1)
            decode1 = self.decode1(decode2,x)
            return decode1
        
class M2UNet_q(nn.Module):
        def __init__(self,encoder,upsamplemode='bilinear',output_channels=1, activation="linear", expand_ratio=0.15):
            super(M2UNet_q,self).__init__()
            encoder = list(encoder.children())[0]
            # Encoder
            self.quant = torch.quantization.QuantStub()

            self.conv1 = encoder[0:2]
            self.conv2 = encoder[2:4]
            self.conv3 = encoder[4:7]
            self.conv4 = encoder[7:14]
            # Decoder
            self.decode4 = DecoderBlock(96,32,upsamplemode,expand_ratio)
            self.decode3 = DecoderBlock(64,24,upsamplemode,expand_ratio)
            self.decode2 = DecoderBlock(44,16,upsamplemode,expand_ratio)
            self.decode1 = LastDecoderBlock(33,upsamplemode,expand_ratio, output_channels=output_channels, activation=activation)
            
            self.dequant = torch.quantization.DeQuantStub()

            # initilaize weights 
            self._init_params()

        def _init_params(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
            
        
        
        def forward(self,x):
            x = self.quant(x)

            conv1 = self.conv1(x)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            decode4 = self.decode4(conv4,conv3)
            decode3 = self.decode3(decode4,conv2)
            decode2 = self.decode2(decode3,conv1)
            decode1 = self.decode1(decode2,x)
            
            res = self.dequant(decode1)

            
            return res
        
def m2unet(output_channels=1,expand_ratio=0.15, activation="linear", **kwargs):
    encoder = Encoder()
    model = M2UNet(encoder,upsamplemode='bilinear',expand_ratio=expand_ratio, output_channels=output_channels, activation=activation)
    return model

def m2unet_q(output_channels=1,expand_ratio=0.15, activation="linear", **kwargs):
    encoder = Encoder()
    model = M2UNet_q(encoder,upsamplemode='bilinear',expand_ratio=expand_ratio, output_channels=output_channels, activation=activation)
    return model


## Defines Unet

import torch
import torchvision
from glob import glob
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transform
from torch.utils.data import DataLoader,Dataset
# from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class Convblock(nn.Module):
    
      def __init__(self,input_channel,output_channel,kernal=3,stride=1,padding=1):
            
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernal,stride,padding),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,kernal),
            nn.ReLU(inplace=True),
        )
    

      def forward(self,x):
        x = self.convblock(x)
        return x
    
class UNet(nn.Module):
    
    def __init__(self,input_channel,retain=True):

        super().__init__()

        self.conv1 = Convblock(input_channel,32)
        self.conv2 = Convblock(32,64)
        self.conv3 = Convblock(64,128)
        self.conv4 = Convblock(128,256)
        self.neck = nn.Conv2d(256,512,3,1)
        self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1)
        self.dconv4 = Convblock(512,256)
        self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        self.dconv3 = Convblock(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.out = nn.Conv2d(32,1,1,1)
        self.retain = retain
        
    def forward(self,x):
        
        # Encoder Network
        
        # Conv down 1
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2,stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2,stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3,kernel_size=2,stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4,kernel_size=2,stride=2)

        # BottelNeck
        neck = self.neck(pool4)
        
        # Decoder Network
        
        # Upconv 1
        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4,upconv4)
        # Making the skip connection 1
        dconv4 = self.dconv4(torch.cat([upconv4,croped],1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3,upconv3)
        # Making the skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3,croped],1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2,upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))
        # Output Layer
        out = self.out(dconv1)
        
        if self.retain == True:
            out = F.interpolate(out,list(x.shape)[2:])

        return out
    
    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)
    
class UNet_q(nn.Module):
    
    def __init__(self,input_channel,retain=True):

        super().__init__()
        
        self.quant = torch.quantization.QuantStub()

        self.conv1 = Convblock(input_channel,32)
        self.conv2 = Convblock(32,64)
        self.conv3 = Convblock(64,128)
        self.conv4 = Convblock(128,256)
        self.neck = nn.Conv2d(256,512,3,1)
        self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1)
        self.dconv4 = Convblock(512,256)
        self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        self.dconv3 = Convblock(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.out = nn.Conv2d(32,1,1,1)
        self.retain = retain
        
        self.dequant = torch.quantization.DeQuantStub()

        
    def forward(self,x):
        
        # Encoder Network
        
        # Conv down 1
        x = self.quant(x)
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2,stride=2)
        # Conv down 2
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2,stride=2)
        # Conv down 3
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3,kernel_size=2,stride=2)
        # Conv down 4
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4,kernel_size=2,stride=2)

        # BottelNeck
        neck = self.neck(pool4)
        
        # Decoder Network
        
        # Upconv 1
        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4,upconv4)
        # Making the skip connection 1
        dconv4 = self.dconv4(torch.cat([upconv4,croped],1))
        # Upconv 2
        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3,upconv3)
        # Making the skip connection 2
        dconv3 = self.dconv3(torch.cat([upconv3,croped],1))
        # Upconv 3
        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2,upconv2)
        # Making the skip connection 3
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        # Upconv 4
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        # Making the skip connection 4
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))
        # Output Layer
        out = self.out(dconv1)
        
        out = self.dequant(dconv1)
        
        if self.retain == True:
            out = F.interpolate(out,list(x.shape)[2:])

        return out
    
    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transform.CenterCrop([H,W])(input_tensor)
    


## Define Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = m2unet().float().to(device)

# Warm up
inp = torch.randn([1, 3, 512, 512]).to(device)
output = model(inp)


## Inference speed
trials = 100
total_time = 0
for i in range(trials):
    inp = torch.randn([1, 3, 512, 512]).to(device)
    torch.cuda.synchronize()
    start_epoch = time.time()
    output = model(inp)
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    total_time += elapsed
    
print('The average time per inference for unquantized m2unet is')
print(total_time/trials)


## Training speed

lr = .0001 # Doesn't matter
lossfunc = nn.MSELoss().cuda('0')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trials = 100
total_time = 0
batch_size = 1
for i in range(trials):
    torch.cuda.synchronize()
                             
    inp = torch.randn([batch_size, 3, 512, 512]).to(device)
    label = torch.randn([batch_size, 1, 512, 512]).to(device)
    start_epoch = time.time()
                                                    
    output = model(inp)                         
    loss = lossfunc(output,label)
    loss.backward()
    optimizer.step()              
                             
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    total_time += elapsed
    
print('The average time per training cycle (batch_size = 1) for unquantized m2unet is')
print(total_time/trials)


'''
Quantization Benchmark
'''

# Need to go back to CPU
device = 'cpu'
# model_fp32 = m2unet_q().to(device)
model_fp32 = UNet_q(3).to(device)

# model must be set to eval mode for static quantization logic to work
model_fp32.eval()

# attach a global qconfig, which contains information about what kind
# of observers to attach. Use 'fbgemm' for server inference and
# 'qnnpack' for mobile inference. Other quantization configurations such
# as selecting symmetric or assymetric quantization and MinMax or L2Norm
# calibration techniques can be specified here.
model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'conv2']])
# model_fp32_fused = model_fp32


# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model_fp32)


# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
input_fp32 = torch.randn(2, 3, 512, 512).to(device)
model_fp32_prepared(input_fp32)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.

        # self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1).
        # self.dconv4 = Convblock(512,256)
        # self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        # self.dconv3 = Convblock(256,128)
        # self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        # self.dconv2 = Convblock(128,64)
        # self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        
# model_fp32_prepared.upconv4.qconfig = None
# model_fp32_prepared.upconv3.qconfig = None

# model_fp32_prepared.upconv2.qconfig = None

# model_fp32_prepared.upconv1.qconfig = None



model_int8 = torch.quantization.convert(model_fp32_prepared)

# run the model, relevant calculations will happen in int8
res = model_int8(input_fp32)

print(res)

# Convert model to Cuda
model_cuda = model_int8.cuda(0)

inp = torch.randn(2, 3, 512, 512).cuda(0)

res = model_cuda(inp)

print('res')

