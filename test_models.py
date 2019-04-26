import argparse
import time
import os
import sys
import json
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from opts import parse_opts
from model import generate_model
from dataset import get_training_set, get_validation_set, get_test_set
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test
from utils import AverageMeter


"""
def calculate_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().data[0]
        ret.append(correct_k / batch_size)

    return ret
"""



def calculate_accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    ret = []
    for k in topk:
        correct_k = correct[:k].float().sum().data[0]
        ret.append(correct_k / batch_size)

    return ret


opt = parse_opts()
if opt.root_path != '':
    opt.video_path = os.path.join(opt.root_path, opt.video_path)
    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
    opt.result_path = os.path.join(opt.root_path, opt.result_path)
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
    if opt.pretrain_path:
        opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
opt.scales = [opt.initial_scale]
for i in range(1, opt.n_scales):
    opt.scales.append(opt.scales[-1] * opt.scale_step)
opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
opt.std = get_std(opt.norm_value)

print(opt)
with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
    json.dump(vars(opt), opt_file)

torch.manual_seed(opt.manual_seed)

model, parameters = generate_model(opt)
print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if
                           p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)


spatial_transform = Compose([
    #Scale(opt.sample_size),
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
    ])
temporal_transform = TemporalCenterCrop(opt.sample_duration)
#temporal_transform = TemporalBeginCrop(opt.sample_duration)
#temporal_transform = TemporalEndCrop(opt.sample_duration)
target_transform = ClassLabel()
validation_data = get_validation_set(
    opt, spatial_transform, temporal_transform, target_transform)
data_loader = torch.utils.data.DataLoader(
    validation_data,
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_threads,
    pin_memory=True)
val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

if opt.resume_path:
    print('loading checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(opt.resume_path)
    assert opt.arch == checkpoint['arch']

    opt.begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

recorder = []

print('run')

model.eval()

batch_time = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()

end_time = time.time()
for i, (inputs, targets) in enumerate(data_loader):
    if not opt.no_cuda:
        targets = targets.cuda(async=True)
    #inputs = Variable(torch.squeeze(inputs), volatile=True)
    inputs = Variable(inputs, volatile=True)
    targets = Variable(targets, volatile=True)
    outputs = model(inputs)
    recorder.append(outputs.data.cpu().numpy().copy())
    #outputs = torch.unsqueeze(torch.mean(outputs, 0), 0)
    prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1, 5))

    top1.update(prec1, inputs.size(0))
    top5.update(prec5, inputs.size(0))

    batch_time.update(time.time() - end_time)
    end_time = time.time()

    print('[{0}/{1}]\t'
          'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
          'prec@1 {top1.avg:.5f} prec@5 {top5.avg:.5f}'.format(
              i + 1,
              len(data_loader),
              batch_time=batch_time,
              top1 =top1,
              top5=top5))

video_pred = [np.argmax(np.mean(x, axis=0)) for x in recorder]
print(video_pred)

with open('annotation_Something/categories.txt') as f:
    lines = f.readlines()
    categories = [item.rstrip() for item in lines]

name_list = [x.strip().split()[0] for x in open('annotation_Something/testlist01.txt')]
order_dict = {e:i for i, e in enumerate(sorted(name_list))}
reorder_output = [None] * len(recorder)
reorder_pred = [None] * len(recorder)
output_csv = []
for i in range(len(recorder)):
    idx = order_dict[name_list[i]]
    reorder_output[idx] = recorder[i]
    reorder_pred[idx] = video_pred[i]
    output_csv.append('%s;%s'%(name_list[i],
                               categories[video_pred[i]]))

    with open('something_predictions.csv','w') as f:
        f.write('\n'.join(output_csv))



print('-----Evaluation is finished------')
print('Overall Prec@1 {:.05f}% Prec@5 {:.05f}%'.format(top1.avg, top5.avg))
