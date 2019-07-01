import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import *
from train import train_epoch
from validation import val_epoch
import test



if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            RandomHorizontalFlip(),
            #RandomRotate(),
            #RandomResize(),
            crop_method,
            #MultiplyValues(),
            #Dropout(),
            #SaltImage(),
            #Gaussian_blur(),
            #SpatialElasticDisplacement(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=16,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])


    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, False, opt)
            
        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, opt)


    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)




