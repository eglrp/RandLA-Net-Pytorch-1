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
from tqdm import tqdm

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
sys.path.append(base_dir)
sys.path.append(root_dir)

from config.config_modelnet import ConfigMODELNET
from net.modelnet_dataset import MODELNET
from net.RandLANet_C import RandLANET, IoUCalculator, compute_loss, compute_acc


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class network:
    def __init__(self, FLAGS):
        self.best_instance_acc = 0.0
        self.best_class_acc = 0.0
        self.mean_correct = []
        self.writer = SummaryWriter('output/modelnet_tensorboard')
        self.f_out = self.mkdir_log(FLAGS.log_dir)
        self.train_dataset = MODELNET('train')
        self.test_dataset = MODELNET('test')
        self.train_dataloader = DataLoaderX(
            self.train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=20,
            worker_init_fn=self.worker_init,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True)
        self.test_dataloader = DataLoaderX(
            self.test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=20,
            worker_init_fn=self.worker_init,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True)
        print('train dataset length:{}'.format(len(self.train_dataset)))
        print('test dataset length:{}'.format(len(self.test_dataset)))
        print('train datalodaer length:{}'.format(len(self.train_dataloader)))
        print('test dataloader length:{}'.format(len(self.test_dataloader)))
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = ConfigMODELNET
        self.net = RandLANET(self.config)
        self.net.to(self.device)
        # torch.cuda.set_device(1)
        # if torch.cuda.device_count() > 1:
        #     log_out("Let's use multi GPUs!", self.f_out)
        #     device_ids=[1,2,3,4]
        #     self.net = nn.DataParallel(self.net, device_ids=[1,2,3,4])
        self.optimizer = optimizer.Adam(self.net.parameters(),
                                        lr=self.config.learning_rate,
                                        betas=(0.9, 0.999),
                                        eps=1e-08,
                                        weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=20,
                                                         gamma=0.7)

        self.end_points = {}
        self.FLAGS = FLAGS

    def mkdir_log(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        f_out = open(os.path.join(out_path, 'log_modelnet_train.txt'), 'a')
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
        # self.adjust_learning_rate(epoch_count)
        self.scheduler.step()
        self.net.train()  # set model to training mode
        for batch_idx, batch_data in tqdm(enumerate(self.train_dataloader),
                                          total=len(self.train_dataloader)):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            xyz = batch_data['xyz']  # (batch,N,3)
            neigh_idx = batch_data['neigh_idx']  # (batch,N,16)
            sub_idx = batch_data['sub_idx']  # (batch,N/4,16)
            interp_idx = batch_data['interp_idx']  # (batch,N,1)
            features = batch_data['features']  # (batch, 3, N)
            labels = batch_data['labels']  # (batch, N)
            input_inds = batch_data['input_inds']  # (batch, N)
            cloud_inds = batch_data['cloud_inds']  # (batch, 1)
            cls = batch_data['cls']

            # Forward pass
            self.optimizer.zero_grad()
            self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx, features,
                                labels, input_inds, cloud_inds)
            self.loss = compute_loss(self.out, cls, self.config)
            pred_choice = self.out.data.max(1)[1]
            correct = pred_choice.eq(cls.long().data).cpu().sum()
            self.mean_correct.append(correct.item() / float(xyz[0].size()[0]))
            self.loss.backward()
            self.optimizer.step()
        train_instance_acc = np.mean(self.mean_correct)
        log_out('Train acc: %f' % train_instance_acc, self.f_out)

    def evaluate_one_epoch(self, epoch_count):
        class_acc = np.zeros((self.config.num_classes, 3))
        self.net.eval()  # set model to eval mode (for bn and dp)
        for batch_idx, batch_data in tqdm(enumerate(self.test_dataloader),
                                          total=len(self.test_dataloader)):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].cuda()
                else:
                    batch_data[key] = batch_data[key].cuda()

            xyz = batch_data['xyz']  # (batch,N,3)
            neigh_idx = batch_data['neigh_idx']  # (batch,N,16)
            sub_idx = batch_data['sub_idx']  # (batch,N/4,16)
            interp_idx = batch_data['interp_idx']  # (batch,N,1)
            features = batch_data['features']  # (batch, 3, N)
            labels = batch_data['labels']  # (batch, N)
            input_inds = batch_data['input_inds']  # (batch, N)
            cloud_inds = batch_data['cloud_inds']  # (batch, 1)
            cls = batch_data['cls']

            # Forward pass
            with torch.no_grad():
                self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx,
                                    features, labels, input_inds, cloud_inds)
                pred_choice = self.out.data.max(1)[1]
                for cat in np.unique(cls.cpu()):
                    classacc = pred_choice[cls == cat].eq(
                        cls[cls == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(
                        xyz[0][cls == cat].size()[0])
                    class_acc[cat, 1] += 1
                correct = pred_choice.eq(cls.long().data).cpu().sum()
                self.mean_correct.append(correct.item() /
                                         float(xyz[0].size()[0]))
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2])
        instance_acc = np.mean(self.mean_correct)

        if (instance_acc >= self.best_instance_acc):
            self.best_instance_acc = instance_acc
        if (class_acc >= self.best_class_acc):
            self.best_class_acc = class_acc
        log_out(
            'Test Instance Accuracy: %f, Class Accuracy: %f' %
            (instance_acc, class_acc), self.f_out)
        log_out(
            'Best Instance Accuracy: %f, Class Accuracy: %f' %
            (self.best_instance_acc, self.best_class_acc), self.f_out)

        try:
            save_dict = self.net.module.state_dict()
        except:
            save_dict = self.net.state_dict()

        # if (instance_acc >= self.best_instance_acc):
        #     state = {
        #         'epoch': epoch_count,
        #         'instance_acc': instance_acc,
        #         'class_acc': class_acc,
        #         'model_state_dict': save_dict,
        #         'optimizer_state_dict': self.optimizer.state_dict(),
        #     }
        #     torch.save(
        #         state,
        #         os.path.join(self.FLAGS.log_dir, 'modelnet_checkpoint.tar'))

    def train(self, start_epoch):
        for epoch in range(start_epoch, self.FLAGS.max_epoch):
            log_out('**************** EPOCH %03d ****************' % (epoch),
                    self.f_out)
            log_out(str(datetime.datetime.now()), self.f_out)
            np.random.seed()
            self.train_one_epoch(epoch)
            self.evaluate_one_epoch(epoch)
            # self.writer.close()

    def run(self):
        start_epoch = 0
        checkpoint_path = self.FLAGS.checkpoint_path
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            log_out(
                "-> loaded checkpoint %s (epoch: %d)" %
                (checkpoint_path, start_epoch), self.f_out)
        self.train(start_epoch)


if __name__ == '__main__':
    writer = SummaryWriter('output/modelnet_tensorboard')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',
                        default='output/modelnet_checkpoint.tar',
                        help='Model checkpoint path [default: None]')
    parser.add_argument(
        '--log_dir',
        default='output',
        help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=ConfigMODELNET.max_epoch,
                        help='Epoch to run [default: 180]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=ConfigMODELNET.batch_size,
                        help='Batch Size during training [default: 8]')
    FLAGS = parser.parse_args()

    network(FLAGS).run()