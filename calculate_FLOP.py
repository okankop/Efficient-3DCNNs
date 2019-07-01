import torch.nn as nn
from thop import profile
from models import squeezenet, shufflenetv2, shufflenet, mobilenet, mobilenetv2, c3d, resnext, resnet

# %%%%%%%%--------------------- SELECT THE MODEL BELOW ---------------------%%%%%%%%

# model = shufflenet.get_model(groups=3, width_mult=0.5, num_classes=600)#1
# model = shufflenetv2.get_model( width_mult=0.25, num_classes=600, sample_size = 112)#2
# model = mobilenet.get_model( width_mult=0.5, num_classes=600, sample_size = 112)#3
# model = mobilenetv2.get_model( width_mult=0.2, num_classes=600, sample_size = 112)#4
# model = shufflenet.get_model(groups=3, width_mult=1.0, num_classes=600)#5
# model = shufflenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#6
# model = mobilenet.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#7
# model = mobilenetv2.get_model( width_mult=0.45, num_classes=600, sample_size = 112)#8
# model = shufflenet.get_model(groups=3, width_mult=1.5, num_classes=600)#9
# model = shufflenetv2.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#10
# model = mobilenet.get_model( width_mult=1.5, num_classes=600, sample_size = 112)#11
# model = mobilenetv2.get_model( width_mult=0.7, num_classes=600, sample_size = 112)#12
# model = shufflenet.get_model(groups=3, width_mult=2.0, num_classes=600)#13
# model = shufflenetv2.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#14
# model = mobilenet.get_model( width_mult=2.0, num_classes=600, sample_size = 112)#15
# model = mobilenetv2.get_model( width_mult=1.0, num_classes=600, sample_size = 112)#16
# model = squeezenet.get_model( version=1.1, num_classes=600, sample_size = 112, sample_duration = 16)
# model = resnext.resnet101( num_classes=600, shortcut_type='B', cardinality=32, sample_size=112, sample_duration=16)
# model = resnet.resnet18( num_classes=600, shortcut_type='A', sample_size=112, sample_duration=16)
# model = resnet.resnet50( num_classes=600, shortcut_type='A', sample_size=112, sample_duration=16)
# model = resnet.resnet101( num_classes=600, shortcut_type='A', sample_size=112, sample_duration=16)
model = c3d.get_model( num_classes=600, sample_size=112, sample_duration=16)
model = model.cuda()
model = nn.DataParallel(model, device_ids=None)	
print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

flops, prms = profile(model, input_size=(1, 3, 16, 112, 112))
print("Total number of FLOPs: ", flops)