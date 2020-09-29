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

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from config.config_semantickitti import Config_SemanticKITTI
from net.semantickitti_dataset import SemanticKITTI
from net.RandLANet import RandLANET, IoUCalculator

def mkdir_log(out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f_out = open(os.path.join(out_path, 'log_train.txt'), 'a')
    return f_out

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

def worker_init(id):
    np.random.seed(np.random.get_state()[1][0] + id)
    
def adjust_lr_rate(optimizer, epoch, confg):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * confg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_one_epoch(net, train_dataloader, optimizer, epoch_count, config, f_out):
    state_dict = {}
    adjust_lr_rate(optimizer, epoch_count, config)
    net.train()
    iou_calculator = IoUCalculator(config)

    for batch_index, batch_data in enumerate(train_dataloader):
        print(batch_index)
        print(batch_data)
    #     for key in batch_data:
    #         if type(batch_data[key]) is list:
    #             for i in range(len(batch_data[key])):
    #                 batch_data[key][i] = batch_data[key][i].cuda()
    #         else:
    #             batch_data[key] = batch_data[key].cuda()
                
    #     # forward
    #     optimizer.zero_grad()
    #     inputs = net(batch_data)
    #     loss, inputs = RandLANET.compute_loss(inputs, config)
    #     loss.backward()
    #     optimizer.step()

    #     accuracy, inputs = RandLANET.compute_accuracy(inputs)
    #     iou_calculator.add_data(inputs)

    #     for key in inputs:
    #         if 'loss' in key or 'accuracy' in key or 'iou' in key:
    #             if key not in state_dict:
    #                 state_dict[key] = 0
    #             state_dict[key] += inputs[key].item()
            
    #     batch_interval = 10
    #     if (batch_index + 1) % batch_interval == 0:
    #         log_out(' ---- batch: %03d ----' % (batch_index + 1), f_out)
    #         for key in sorted(state_dict.keys()):
    #             log_out('mean %s: %f' % (key, state_dict[key] / batch_interval), f_out)
    #             stat_dict[key] = 0
    # mean_iou, iou_list = iou_calculator.compute_iou()
    # log_out('mean IoU:{:.1f}'.format(mean_iou * 100), f_out)
    # s = 'IoU:'
    # for iou_tmp in iou_list:
    #     s += '{:5.2f} '.format(100 * iou_tmp)
    # log_out(s, f_out)


def evaluate_one_epoch(net, test_dataloader, config, f_out):
    state_dict = {} 
    net.eval()
    iou_calculator = IoUCalculator(config)
    for batch_index, batch_data in enumerate(test_dataloader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].cuda()
            else:
                batch_data[key] = batch_data[key].cuda()

        # Forward pass
        with torch.no_grad():
            inputs = net(batch_data)

        loss, inputs = RandLANET.compute_loss(inputs, config)

        accuracy, inputs = RandLANET.compute_accuracy(inputs)
        iou_calculator.add_data(inputs)

        # Accumulate statistics and print out
        for key in inputs:
            if 'loss' in key or 'accuracy' in key or 'iou' in key:
                if key not in state_dict:
                    state_dict[key] = 0
                state_dict[key] += inputs[key].item()

        batch_interval = 10
        if (batch_index + 1) % batch_interval == 0:
            log_out(' ---- batch: %03d ----' % (batch_index + 1), f_out)

    for key in sorted(state_dict.keys()):
        log_out('eval mean %s: %f'%(key, state_dict[key]/(float(batch_index+1))))
    mean_iou, iou_list = iou_calculator.compute_iou()
    log_out('mean IoU:{:.1f}'.format(mean_iou * 100), f_out)
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_out(s, f_out)
    
def train(net, train_dataloader, test_dataloader, optimizer, config, start_epoch, flags, f_out):
    loss = 0
    for epoch in range(start_epoch, flags.max_epoch):
        epoch_count = epoch
        log_out('**** EPOCH %03d ****' % (epoch), f_out)
        log_out(str(datetime.datetime.now()), f_out)
        np.random.seed()
        train_one_epoch(net, train_dataloader, optimizer, epoch_count, config, f_out)

        # if epoch_count == 0 or epoch_count % 10 == 9:
        #     log_out('**** EVAL EPOCH %03d START****' % (epoch), f_out)
        #     evaluate_one_epoch(net, test_dataloader, config, f_out)
        #     log_out('**** EVAL EPOCH %03d END****' % (epoch), f_out)
        
        # save_dict = {
        #     'epoch': epoch + 1,
        #     'optimzer_state_dict': optimizer.state_dict(),
        #     'loss': loss
        # }

        # try:
        #     save_dict['model_state_dict'] = net.module.state_dict()
        # except:
        #     save_dict['model_state_dict'] = net.state_dict()
        # torch.save(save_dict, os.path.join(flags.log_dir, 'checkpoint.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='output/checkpoint.tar', help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 8]')
    FLAGS = parser.parse_args()

    f_out = mkdir_log(FLAGS.log_dir)

    train_dataset = SemanticKITTI('training')
    test_dataset = SemanticKITTI('validation')
    print('train dataset length:{}'.format(len(train_dataset)))
    print('test dataset length:{}'.format(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=worker_init, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=worker_init, collate_fn=test_dataset.collate_fn)
    print('train datalodaer length:{}'.format(len(train_dataloader)))
    print('test dataloader length:{}'.format(len(test_dataloader)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = RandLANET('SemanticKITTI', Config_SemanticKITTI)
    net.to(device)
    if torch.cuda.device_count() > 1:
        log_out("Let's use multi GPUs!")
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
    optimizer = optimizer.Adam(net.parameters(), lr=Config_SemanticKITTI.learning_rate)

    it = -1
    start_epoch = 0
    checkpoint_path = FLAGS.checkpoint_path
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log_out("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))
    
    train(net, train_dataloader, test_dataloader, optimizer, Config_SemanticKITTI, start_epoch, FLAGS, f_out)



