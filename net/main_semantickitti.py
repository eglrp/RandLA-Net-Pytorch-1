#! ~/.miniconda3/envs/pytorch/bin/python

import os
import sys
import time
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from prefetch_generator import BackgroundGenerator


base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from config.config_semantickitti import ConfigSemanticKITTI
from net.semantickitti_dataset import SemanticKITTI
from net.RandLANet import RandLANET, IoUCalculator, compute_loss, compute_acc

def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class network:
    def __init__(self, FLAGS):
        self.writer = SummaryWriter('output/semantickitti_tensorboard')
        self.f_out = self.mkdir_log(FLAGS.log_dir)
        self.train_dataset = SemanticKITTI('training', FLAGS.test_area)
        self.test_dataset = SemanticKITTI('validation', FLAGS.test_area)
        print('train dataset length:{}'.format(len(self.train_dataset)))
        print('test dataset length:{}'.format(len(self.test_dataset)))
        self.train_dataloader = DataLoaderX(self.train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=self.worker_init, collate_fn=self.train_dataset.collate_fn, pin_memory=True)
        self.test_dataloader = DataLoaderX(self.test_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=20, worker_init_fn=self.worker_init, collate_fn=self.test_dataset.collate_fn, pin_memory=True)
        print('train datalodaer length:{}'.format(len(self.train_dataloader)))
        print('test dataloader length:{}'.format(len(self.test_dataloader)))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = ConfigSemanticKITTI
        self.net = RandLANET('SemanticKITTI', self.config )
        self.net.to(self.device)
        # torch.cuda.set_device(1) 
        # if torch.cuda.device_count() > 1:
        #     log_out("Let's use multi GPUs!", self.f_out)
        #     device_ids=[1,2,3,4]
        #     self.net = nn.DataParallel(self.net, device_ids=[1,2,3,4])
        self.optimizer = optimizer.Adam(self.net.parameters(), lr=self.config .learning_rate)

        self.end_points = {}
        self.FLAGS = FLAGS

    def mkdir_log(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        f_out = open(os.path.join(out_path, 'log_semantickitti_train.txt'), 'a')
        return f_out

    def worker_init(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def adjust_learning_rate(self, epoch):
        lr = self.optimizer.param_groups[0]['lr']
        lr = lr * self.config.lr_decays[epoch]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.writer.add_scalar('learning rate', lr, epoch)

    def train_one_epoch(self, epoch_count):
        self.stat_dict = {}  # collect statistics
        self.adjust_learning_rate(epoch_count)
        self.net.train()  # set model to training mode
        iou_calc = IoUCalculator(self.config)
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            t_start = time.time()
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            xyz = batch_data['xyz'] # (batch,N,3)
            neigh_idx = batch_data['neigh_idx'] # (batch,N,16)
            sub_idx = batch_data['sub_idx']  # (batch,N/4,16)
            interp_idx = batch_data['interp_idx'] # (batch,N,1)
            features = batch_data['features'] # (batch, 3, N)
            labels = batch_data['labels']  # (batch, N)
            input_inds = batch_data['input_inds'] # (batch, N)
            cloud_inds = batch_data['cloud_inds'] # (batch, 1)

            # Forward pass
            self.optimizer.zero_grad()
            self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx, features, labels, input_inds, cloud_inds)
            self.loss, self.end_points['valid_logits'], self.end_points['valid_labels'] = compute_loss(self.out, labels, self.config)
            self.end_points['loss'] = self.loss
            # self.writer.add_graph(self.net, input_to_model=[xyz, neigh_idx, sub_idx, interp_idx, features, labels, input_inds, cloud_inds])
            self.writer.add_scalar('training loss', self.loss, (epoch_count * len(self.train_dataloader) + batch_idx))
            
            self.loss.backward()
            self.optimizer.step()

            self.acc = compute_acc(self.end_points['valid_logits'], self.end_points['valid_labels'])
            self.end_points['acc'] = self.acc
            self.writer.add_scalar('training accuracy', self.acc, (epoch_count * len(self.train_dataloader) + batch_idx))
            iou_calc.add_data(self.end_points['valid_logits'], self.end_points['valid_labels'])

            for key in self.end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:
                    if key not in self.stat_dict:
                        self.stat_dict[key] = 0
                    self.stat_dict[key] += self.end_points[key].item()
            t_end = time.time()
                
            batch_interval = 10
            if (batch_idx + 1) % batch_interval == 0:
                log_out(' ----step %08d batch: %08d ----' %(epoch_count * len(self.train_dataloader) + batch_idx+1,  (batch_idx + 1)), self.f_out)
                for key in sorted(self.stat_dict.keys()):
                    log_out('mean %s: %f---%f ms' % (key, self.stat_dict[key] / batch_interval, 1000 * (t_end - t_start)), self.f_out)
                    self.writer.add_scalar('training mean {}'.format(key), self.stat_dict[key] / batch_interval, (epoch_count * len(self.train_dataloader) + batch_idx))
                    self.stat_dict[key] = 0

            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name + '_grad', param.grad, (epoch_count * len(self.train_dataloader) + batch_idx))
                self.writer.add_histogram(name + '_data', param, (epoch_count * len(self.train_dataloader) + batch_idx))
        mean_iou, iou_list = iou_calc.compute_iou()
        self.writer.add_scalar('training mean iou', mean_iou, (epoch_count * len(self.train_dataloader)))
        log_out('training mean IoU:{:.1f}'.format(mean_iou * 100), self.f_out)
        s = 'training IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_out(s, self.f_out)
        self.writer.close()

    def evaluate_one_epoch(self, epoch_count):
        self.current_loss = None
        self.net.eval() # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(self.config)
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            t_start = time.time()
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            xyz = batch_data['xyz'] # (batch,N,3)
            neigh_idx = batch_data['neigh_idx'] # (batch,N,16)
            sub_idx = batch_data['sub_idx']  # (batch,N/4,16)
            interp_idx = batch_data['interp_idx'] # (batch,N,1)
            features = batch_data['features'] # (batch, 3, N)
            labels = batch_data['labels']  # (batch, N)
            input_inds = batch_data['input_inds'] # (batch, N)
            cloud_inds = batch_data['cloud_inds'] # (batch, 1)

            # Forward pass
            with torch.no_grad():
                self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx, features, labels, input_inds, cloud_inds)

            self.loss, self.end_points['valid_logits'], self.end_points['valid_labels'] = compute_loss(self.out, labels, self.config)
            self.end_points['loss'] = self.loss
            # self.writer.add_scalar('eval loss', self.loss, (epoch_count* len(self.test_dataloader) + batch_idx))
            self.acc = compute_acc(self.end_points['valid_logits'], self.end_points['valid_labels'])
            self.end_points['acc'] = self.acc
            # self.writer.add_scalar('eval acc', self.acc, (epoch_count* len(self.test_dataloader) + batch_idx))
            iou_calc.add_data(self.end_points['valid_logits'], self.end_points['valid_labels'])

            # Accumulate statistics and print out
            for key in self.end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:
                    if key not in self.stat_dict:
                        self.stat_dict[key] = 0
                    self.stat_dict[key] += self.end_points[key].item()

            t_end = time.time()

            batch_interval = 10
            if (batch_idx + 1) % batch_interval == 0:
                log_out(' ----step %08d batch: %08d ----' %(epoch_count * len(self.test_dataloader) + batch_idx+1,  (batch_idx + 1)), self.f_out)

        for key in sorted(self.stat_dict.keys()):
            log_out('mean %s: %f---%f ms' % (key, self.stat_dict[key] / batch_interval, 1000 * (t_end - t_start)), self.f_out)
            self.writer.add_scalar('eval mean {}'.format(key), self.stat_dict[key] / (float(batch_idx + 1)), (epoch_count * len(self.test_dataloader)))
        mean_iou, iou_list = iou_calc.compute_iou()
        self.writer.add_scalar('eval mean iou', mean_iou, (epoch_count * len(self.test_dataloader)))
        log_out('eval mean IoU:{:.1f}'.format(mean_iou * 100), self.f_out)
        s = 'eval IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_out(s, self.f_out)
        self.writer.close()

        current_loss = self.stat_dict['loss'] / (float(batch_idx + 1))
        return current_loss
    
    def train(self,start_epoch):
        loss = 0
        min_loss = 100
        current_loss = None
        for epoch in range(start_epoch, self.FLAGS.max_epoch):
            log_out('**************** EPOCH %03d ****************' % (epoch), self.f_out)
            log_out(str(datetime.datetime.now()), self.f_out)
            np.random.seed()
            self.train_one_epoch(epoch)

            if epoch == 0 or epoch % 10 == 9:
                log_out('**** EVAL EPOCH %03d START****' % (epoch), self.f_out)
                current_loss = self.evaluate_one_epoch(epoch)
                log_out('**** EVAL EPOCH %03d END****' % (epoch), self.f_out)
            
            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        }

            try:
                save_dict['model_state_dict'] = self.net.module.state_dict()
            except:
                save_dict['model_state_dict'] = self.net.state_dict()
            
            torch.save(save_dict, os.path.join(self.FLAGS.log_dir, 'semantickitti_checkpoint.tar'))

    def run(self):
        it = -1
        start_epoch = 0
        checkpoint_path = self.FLAGS.checkpoint_path
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            log_out("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch), self.f_out)
        self.train(start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='output/semantickitti_checkpoint.tar', help='Model checkpoint path [default: None]')
    parser.add_argument('--log_dir', default='output', help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch', type=int, default=ConfigSemanticKITTI.max_epoch, help='Epoch to run [default: 180]')
    parser.add_argument('--batch_size', type=int, default=ConfigSemanticKITTI.batch_size, help='Batch Size during training [default: 8]')
    parser.add_argument('--test_area', type=str, default='14', help='options: 08, 11,12,13,14,15,16,17,18,19,20,21')
    FLAGS = parser.parse_args()

    network(FLAGS).run()



