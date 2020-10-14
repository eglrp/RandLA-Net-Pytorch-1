#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from config.config_semantic3d import ConfigSemantic3D
from net.semanti3d_dataset import Semantic3D
from net.RandLANet import RandLANET, IoUCalculator, compute_loss, compute_acc

def mkdir_log(out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f_out = open(os.path.join(out_path, 'log_semantic3d_train.txt'), 'a')
    return f_out

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def worker_init(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def adjust_learning_rate(optimizer, epoch, config, writer):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * config.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    writer.add_scalar('learning rate', lr, epoch * config.batch_size)

def train_one_epoch(net, train_dataloader, optimizer, epoch_count, config, f_out, writer):
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, epoch_count, config, writer)
    net.train()  # set model to training mode
    iou_calc = IoUCalculator(config)
    for batch_idx, batch_data in enumerate(train_dataloader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        optimizer.zero_grad()
        end_points = net(batch_data)
        loss, end_points = compute_loss(end_points, config)
        writer.add_scalar('training loss', loss, (epoch_count * len(train_dataloader) + batch_idx) * config.batch_size)
        
        loss.backward()
        optimizer.step()

        acc, end_points = compute_acc(end_points)
        writer.add_scalar('training accuracy', acc, (epoch_count * len(train_dataloader) + batch_idx)*config.batch_size)
        iou_calc.add_data(end_points)

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
            
        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_out(' ---- batch: %03d ----' % (batch_idx + 1), f_out)
            for key in sorted(stat_dict.keys()):
                log_out('mean %s: %f' % (key, stat_dict[key] / batch_interval), f_out)
                writer.add_scalar('training mean %s'%(key), stat_dict[key] / batch_interval, (epoch_count * len(train_dataloader) + batch_idx)*config.batch_size)
                stat_dict[key] = 0

        for name, param in net.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, (epoch_count * len(train_dataloader) + batch_idx)*config.batch_size)
            writer.add_histogram(name + '_data', param, (epoch_count * len(train_dataloader) + batch_idx)*config.batch_size)
        writer.flush()
    mean_iou, iou_list = iou_calc.compute_iou()
    writer.add_scalar('training mean iou', mean_iou, (epoch_count * len(train_dataloader))*config.batch_size)
    log_out('mean IoU:{:.1f}'.format(mean_iou * 100), f_out)
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_out(s, f_out)
    writer.flush()
    writer.close()


def evaluate_one_epoch(net, test_dataloader, epoch_count, config, f_out):
    stat_dict = {} # collect statistics
    net.eval() # set model to eval mode (for bn and dp)
    iou_calc = IoUCalculator(config)
    for batch_idx, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)

        loss, end_points = compute_loss(end_points, config)
        writer.add_scalar('eval loss', loss, (epoch_count* len(test_dataloader) + batch_idx)*config.batch_size)
        acc, end_points = compute_acc(end_points)
        writer.add_scalar('eval acc', acc, (epoch_count* len(test_dataloader) + batch_idx)*config.batch_size)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx + 1) % batch_interval == 0:
            log_out(' ---- batch: %03d ----' % (batch_idx + 1), f_out)
        writer.flush()

    for key in sorted(stat_dict.keys()):
        log_out('eval mean %s: %f' % (key, stat_dict[key] / (float(batch_idx + 1))), f_out)
        # writer.add_scalar('eval mean %s'% (key), stat_dict[key] / (float(batch_idx + 1)), (epoch * len(train_dataloader))*config.batch_size)
    mean_iou, iou_list = iou_calc.compute_iou()
    # writer.add_scalar('eval mean iou', mean_iou, (epoch * len(train_dataloader))*config.batch_size)
    log_out('mean IoU:{:.1f}'.format(mean_iou * 100), f_out)
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_out(s, f_out)
    writer.flush()
    writer.close()

def train(net, train_dataloader, test_dataloader, optimizer, config, start_epoch, flags, f_out, writer):
    loss = 0
    for epoch in range(start_epoch, FLAGS.max_epoch):
        log_out('**** EPOCH %03d ****' % (epoch), f_out)
        log_out(str(datetime.datetime.now()), f_out)
        np.random.seed()
        train_one_epoch(net, train_dataloader, optimizer, epoch, config, f_out, writer)

        if epoch == 0 or epoch % 10 == 9:
            log_out('**** EVAL EPOCH %03d START****' % (epoch), f_out)
            evaluate_one_epoch(net, test_dataloader, epoch, config, f_out)
            log_out('**** EVAL EPOCH %03d END****' % (epoch), f_out)
        
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }

        try:
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(flags.log_dir, 'semantic3d_checkpoint.tar'))

if __name__ == '__main__':
    writer = SummaryWriter('output/semantic3d_tensorboard')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='output/semantic3d_checkpoint.tar', help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 8]')
    FLAGS = parser.parse_args()

    f_out = mkdir_log(FLAGS.log_dir)

    train_dataset = Semantic3D('training')
    test_dataset = Semantic3D('validation')
    # print('train dataset length:{}'.format(len(train_dataset)))
    # print('test dataset length:{}'.format(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1, worker_init_fn=worker_init, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=1, worker_init_fn=worker_init, collate_fn=test_dataset.collate_fn)
    # print('train datalodaer length:{}'.format(len(train_dataloader)))
    # print('test dataloader length:{}'.format(len(test_dataloader)))

    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    net = RandLANET('Semantic3D', ConfigSemantic3D)
    # print(net)
    net.to(device)
    torch.cuda.set_device(1) 
    if torch.cuda.device_count() > 1:
        log_out("Let's use multi GPUs!", f_out)
        net = nn.DataParallel(net, device_ids=[1,2,3,4])
    optimizer = optimizer.Adam(net.parameters(), lr=ConfigSemantic3D.learning_rate)

    it = -1
    start_epoch = 0
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_out("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch), f_out)
    
    train(net, train_dataloader, test_dataloader, optimizer, ConfigSemantic3D, start_epoch, FLAGS, f_out, writer)

