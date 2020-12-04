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
        self.train_dataloader = DataLoaderX(
            self.train_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.batch_size,
            worker_init_fn=self.worker_init,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True)
        self.test_dataloader = DataLoaderX(
            self.test_dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.batch_size,
            worker_init_fn=self.worker_init,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True)
        print('train datalodaer length:{}'.format(len(self.train_dataloader)))
        print('test dataloader length:{}'.format(len(self.test_dataloader)))
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.config = ConfigSemanticKITTI
        self.net = RandLANET('SemanticKITTI', self.config)
        self.net.to(self.device)
        torch.cuda.set_device(1)
        if torch.cuda.device_count() > 1:
            log_out("Let's use multi GPUs!", self.f_out)
            self.net = nn.DataParallel(self.net, device_ids=[1, 2])
        self.optimizer = optimizer.Adam(self.net.parameters(),
                                        lr=self.config.learning_rate,
                                        weight_decay=0.0001)

        self.FLAGS = FLAGS
        self.end_points = {}
        self.stat_dict = {}
        self.best_iou = [0]
        self.mIou_list = [0]

    def mkdir_log(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        f_out = open(os.path.join(out_path, 'log_semantickitti_train.txt'),
                     'a')
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
        self.adjust_learning_rate(epoch_count)
        self.net.train()
        iou_calc = IoUCalculator(self.config)
        for batch_idx, batch_data in enumerate(self.train_dataloader):
            t_start = time.time()
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

            # Forward pass
            self.optimizer.zero_grad()
            self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx, features,
                                labels, input_inds, cloud_inds)

            self.loss, self.end_points['valid_logits'], self.end_points[
                'valid_labels'] = compute_loss(self.out, labels, self.config)
            self.end_points['loss'] = self.loss

            self.loss.backward()
            self.optimizer.step()

            self.acc = compute_acc(self.end_points['valid_logits'],
                                   self.end_points['valid_labels'])
            self.end_points['acc'] = self.acc
            iou_calc.add_data(self.end_points['valid_logits'],
                              self.end_points['valid_labels'])
            t_end = time.time()

            self.writer.add_scalar(
                'Loss/Train', self.loss,
                (epoch_count * len(self.train_dataloader) + batch_idx + 1))
            self.writer.add_scalar(
                'Acc/Train', self.acc,
                (epoch_count * len(self.train_dataloader) + batch_idx + 1))

            for key in self.end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:
                    if key not in self.stat_dict:
                        self.stat_dict[key] = 0
                    self.stat_dict[key] += self.end_points[key].item()

            batch_interval = 50
            if (batch_idx + 1) % batch_interval == 0:
                message = 'Step {:08d} L_out={:5.6f} Acc={:4.6f} ' '---{:8.2f} ms/batch'
                loss_avg = 0
                acc_avg = 0
                loss_avg = self.stat_dict['loss'] / batch_interval
                acc_avg = self.stat_dict['acc'] / batch_interval
                self.stat_dict['loss'] = 0
                self.stat_dict['acc'] = 0
                log_out(
                    message.format(
                        epoch_count * len(self.train_dataloader) + batch_idx +
                        1, loss_avg, acc_avg, 1000 * (t_end - t_start)),
                    self.f_out)

                self.writer.add_scalar(
                    'Mean Loss/Train', loss_avg,
                    (epoch_count * len(self.train_dataloader) + batch_idx + 1))
                self.writer.add_scalar(
                    'Mean Acc/Train', acc_avg,
                    (epoch_count * len(self.train_dataloader) + batch_idx + 1))

        mean_iou, iou_list = iou_calc.compute_iou()
        # tensorboard
        self.writer.add_scalar(
            'mean_iou/Train', mean_iou * 100,
            ((epoch_count + 1) * len(self.train_dataloader)))
        s = 'train IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_out(s, self.f_out)
        self.writer.close()

    def evaluate_one_epoch(self, epoch_count):
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(self.config)
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            t_start = time.time()
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

            # Forward pass
            with torch.no_grad():
                self.out = self.net(xyz, neigh_idx, sub_idx, interp_idx,
                                    features, labels, input_inds, cloud_inds)

            self.loss, self.end_points['valid_logits'], self.end_points[
                'valid_labels'] = compute_loss(self.out, labels, self.config)
            self.end_points['loss'] = self.loss
            t_end = time.time()

            self.acc = compute_acc(self.end_points['valid_logits'],
                                   self.end_points['valid_labels'])
            self.end_points['acc'] = self.acc
            iou_calc.add_data(self.end_points['valid_logits'],
                              self.end_points['valid_labels'])

            self.writer.add_scalar(
                'Loss/Test', self.loss,
                (epoch_count * len(self.test_dataloader) + batch_idx + 1))
            self.writer.add_scalar(
                'Acc/Test', self.acc,
                (epoch_count * len(self.test_dataloader) + batch_idx + 1))

            for key in self.end_points:
                if 'loss' in key or 'acc' in key or 'iou' in key:
                    if key not in self.stat_dict:
                        self.stat_dict[key] = 0
                    self.stat_dict[key] += self.end_points[key].item()

            batch_interval = 50
            if (batch_idx + 1) % batch_interval == 0:
                message = 'Step {:08d} L_out={:5.6f} Acc={:4.6f} ' '---{:8.2f} ms/batch'
                loss_avg = 0
                acc_avg = 0
                loss_avg = self.stat_dict['loss'] / batch_interval
                acc_avg = self.stat_dict['acc'] / batch_interval
                self.stat_dict['loss'] = 0
                self.stat_dict['acc'] = 0
                log_out(
                    message.format(
                        epoch_count * len(self.test_dataloader) + batch_idx +
                        1, loss_avg, acc_avg, 1000 * (t_end - t_start)),
                    self.f_out)

                self.writer.add_scalar(
                    'Mean Loss/Test', loss_avg,
                    (epoch_count * len(self.test_dataloader) + batch_idx + 1))
                self.writer.add_scalar(
                    'Mean Acc/Test', acc_avg,
                    (epoch_count * len(self.test_dataloader) + batch_idx + 1))

        mean_iou, iou_list = iou_calc.compute_iou()
        log_out('eval mean IoU:{:.1f}'.format(mean_iou * 100), self.f_out)
        self.writer.add_scalar('mean_iou/Test', mean_iou * 100,
                               ((epoch_count + 1) * len(self.test_dataloader)))
        s = 'eval IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        log_out(s, self.f_out)
        self.writer.close()

        return mean_iou

    def train(self, start_epoch):
        for epoch in range(start_epoch, self.FLAGS.max_epoch):
            log_out(
                '****************TRAIN EPOCH %03d START****************' %
                (epoch), self.f_out)
            log_out(str(datetime.datetime.now()), self.f_out)
            self.train_one_epoch(epoch)

            log_out(
                '****************EVAL EPOCH %03d START****************' %
                (epoch), self.f_out)
            log_out(str(datetime.datetime.now()), self.f_out)
            mean_iou = self.evaluate_one_epoch(epoch)

            self.save_params(self.best_iou, mean_iou, epoch)

    def save_params(self, best_iou, current_iou, epoch):
        save_dict = {
            'epoch': epoch +
            1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except:
            save_dict['model_state_dict'] = self.net.state_dict()

        current_iou = float(current_iou)
        if current_iou > best_iou[0]:
            best_iou[0] = current_iou
            # net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
            torch.save(
                save_dict,
                os.path.join(self.FLAGS.log_dir,
                             'semantickitti_checkpoint_best.tar'))
            log_out('save semantickitti_checkpoint_best.tar', self.f_out)
        else:
            torch.save(
                save_dict,
                os.path.join(
                    self.FLAGS.log_dir,
                    'semantickitti_checkpoint_{}_{}.tar'.format(
                        epoch, current_iou)))
            log_out(
                'save semantickitti_checkpoint_{}_{}.tar'.format(
                    epoch, current_iou), self.f_out)

    def run(self):
        start_epoch = 0
        checkpoint_path = self.FLAGS.checkpoint_path
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            log_out(
                "-> loaded checkpoint %s (epoch: %d)" %
                (checkpoint_path, start_epoch), self.f_out)
        self.train(start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path',
                        default='output/semantickitti_checkpoint.tar',
                        help='Model checkpoint path [default: None]')
    parser.add_argument(
        '--log_dir',
        default='output',
        help='Dump dir to save model checkpoint [default: log]')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=ConfigSemanticKITTI.max_epoch,
                        help='Epoch to run [default: 180]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=ConfigSemanticKITTI.batch_size,
                        help='Batch Size during training [default: 8]')
    parser.add_argument('--test_area',
                        type=str,
                        default='14',
                        help='options: 08, 11,12,13,14,15,16,17,18,19,20,21')
    FLAGS = parser.parse_args()

    network(FLAGS).run()
