import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb

from utils import AverageMeter, calculate_accuracy
from models import squeezenet, shufflenetv2, shufflenet, mobilenet, mobilenetv2, c3d, resnet, resnext

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
# model = squeezenet.get_model( version=1.1, num_classes=600, sample_size = 112, sample_duration = 8)
# model = resnet.resnet18(sample_size = 112, sample_duration = 8, num_classes=600)
# model = resnet.resnet50(sample_size = 112, sample_duration = 8, num_classes=600)
# model = resnet.resnet101(sample_size = 112, sample_duration = 8, num_classes=600)
model = resnext.resnext101(sample_size = 112, sample_duration = 8, num_classes=600)

model = model.cuda()
model = nn.DataParallel(model, device_ids=None)	
print(model)

batch_time = AverageMeter()
input_var = Variable(torch.randn(8, 3, 16, 112, 112))
end_time = time.time()

for i in range(1000):

	output = model(input_var)
	batch_time.update(time.time() - end_time)
	end_time = time.time()
	print("Current average time: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 8.0) )

print("Average time for CPU: ", batch_time.avg, "Speed (vps): ", 1 / (batch_time.avg / 8.0))
