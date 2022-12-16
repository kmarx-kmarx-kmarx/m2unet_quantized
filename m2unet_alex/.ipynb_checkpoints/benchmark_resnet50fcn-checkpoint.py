import torch
import torch.nn as nn
import math


'''
The purpose of this file is to benchmark the speeds of different models before and after quantization
'''
import time

import torch

device = 'cuda:0'

model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device)
device = 'cuda:0'



# # Define Model

device = 'cuda:0'

model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device)
device = 'cuda:0'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

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

print('The average time per inference for unquantized unet is')
print(total_time/trials)


# # Training speed

lr = .0001 # Doesn't matter
lossfunc = nn.MSELoss().cuda('0')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

trials = 100
total_time = 0
batch_size = 1
for i in range(trials):
    torch.cuda.synchronize()
                             
    inp = torch.randn([batch_size, 3, 512, 512]).to(device)
    inp2 = torch.randn([batch_size, 1, 512, 512]).to(device)
    
    # Create artifical label

    # label = torch.randn([batch_size, 1, 512, 512]).to(device)
    start_epoch = time.time()
                                                    
    output = model(inp)                         
    loss = lossfunc(output['out'],inp2)
    loss.backward()
    optimizer.step()              
                             
    torch.cuda.synchronize()
    end_epoch = time.time()
    elapsed = end_epoch - start_epoch
    total_time += elapsed

print('The average time per training cycle (batch_size = 1) for unquantized unet is')
print(total_time/trials)



'''
Quantization Benchmark
'''

# # Need to go back to CPU
# device = 'cpu'
# # model_fp32 = m2unet_q().to(device)
# model_fp32 = UNet_q(3).to(device)

# # model must be set to eval mode for static quantization logic to work
# model_fp32.eval()

# # attach a global qconfig, which contains information about what kind
# # of observers to attach. Use 'fbgemm' for server inference and
# # 'qnnpack' for mobile inference. Other quantization configurations such
# # as selecting symmetric or assymetric quantization and MinMax or L2Norm
# # calibration techniques can be specified here.
# model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')

# # Fuse the activations to preceding layers, where applicable.
# # This needs to be done manually depending on the model architecture.
# # Common fusions include `conv + relu` and `conv + batchnorm + relu`
# # model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv1', 'conv2']])
# # model_fp32_fused = model_fp32


# # Prepare the model for static quantization. This inserts observers in
# # the model that will observe activation tensors during calibration.
# model_fp32_prepared = torch.quantization.prepare(model_fp32)


# # calibrate the prepared model to determine quantization parameters for activations
# # in a real world setting, the calibration would be done with a representative dataset
# input_fp32 = torch.randn(2, 3, 512, 512).to(device)
# model_fp32_prepared(input_fp32)

# # Convert the observed model to a quantized model. This does several things:
# # quantizes the weights, computes and stores the scale and bias value to be
# # used with each activation tensor, and replaces key operators with quantized
# # implementations.

#         # self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1).
#         # self.dconv4 = Convblock(512,256)
#         # self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
#         # self.dconv3 = Convblock(256,128)
#         # self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
#         # self.dconv2 = Convblock(128,64)
#         # self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)

# # model_fp32_prepared.upconv4.qconfig = None
# # model_fp32_prepared.upconv3.qconfig = None

# # model_fp32_prepared.upconv2.qconfig = None

# # model_fp32_prepared.upconv1.qconfig = None



# model_int8 = torch.quantization.convert(model_fp32_prepared)

# # run the model, relevant calculations will happen in int8
# res = model_int8(input_fp32)

# print(res)

# # Convert model to Cuda
# model_cuda = model_int8.cuda(0)

# inp = torch.randn(2, 3, 512, 512).cuda(0)

# res = model_cuda(inp)

# print('res')

