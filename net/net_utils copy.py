#! ~/.miniconda3/envs/pytorch/bin/python
import torch
import torch.nn as nn
from typing import List, Tuple


class conv1d(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 name,
                 bias=True,
                 activation_fn=nn.ReLU(inplace=True),
                 bn=True,
                 eps=1e-6,
                 momentum=0.99):
        super(conv1d, self).__init__()
        conv1d_layer = nn.Conv1d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)
        nn.init.xavier_normal_(conv1d_layer.weight.data)
        if bias is True:
            nn.init.constant_(conv1d_layer.bias.data, 0)
        self.add_module(name + 'conv1d', conv1d_layer)

        if bn is True:
            bn_layer = nn.BatchNorm1d(num_features=out_channels,
                                      eps=eps,
                                      momentum=momentum)
            self.add_module(name + 'bn1d', bn_layer)
        if activation_fn is not None:
            self.add_module(name + 'activation', activation_fn)


class conv2d(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 name,
                 bias=True,
                 activation_fn=nn.ReLU(inplace=True),
                 bn=True,
                 eps=1e-6,
                 momentum=0.99):
        super(conv2d, self).__init__()
        conv2d_layer = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 bias=bias)
        nn.init.xavier_normal_(conv2d_layer.weight.data)
        if bias is True:
            nn.init.constant_(conv2d_layer.bias.data, 0)
        self.add_module(name + 'conv2d', conv2d_layer)

        if bn is True:
            bn_layer = nn.BatchNorm2d(num_features=out_channels,
                                      eps=eps,
                                      momentum=momentum)
            self.add_module(name + 'bn2d', bn_layer)

        if activation_fn is not None:
            self.add_module(name + 'activation', activation_fn)


class conv2d_transpose(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 name,
                 bias=True,
                 activation_fn=nn.ReLU(inplace=True),
                 bn=True,
                 eps=1e-6,
                 momentum=0.99):
        super(conv2d_transpose, self).__init__()
        conv2d_transpose_layer = nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=bias)
        nn.init.xavier_normal_(conv2d_transpose_layer.weight.data)
        if bias is True:
            nn.init.constant_(conv2d_transpose_layer.bias.data, 0)
        self.add_module(name + 'conv2d_transpose', conv2d_transpose_layer)

        if bn is True:
            bn_layer = nn.BatchNorm2d(num_features=out_channels,
                                      eps=eps,
                                      momentum=momentum)
            self.add_module(name + 'bn2d', bn_layer)

        if activation_fn is not None:
            self.add_module(name + 'activation', activation_fn)


class SharedMLP(nn.Sequential):
    def __init__(self,
                 args: List[int],
                 *,
                 bn: bool = False,
                 activation=nn.ReLU(inplace=True),
                 preact: bool = False,
                 first: bool = False,
                 name: str = "",
                 instance_norm: bool = False):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(args[i],
                       args[i + 1],
                       bn=(not first or not preact or (i != 0)) and bn,
                       activation=activation if
                       (not first or not preact or (i != 0)) else None,
                       preact=preact,
                       instance_norm=instance_norm))


class _ConvBase(nn.Sequential):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size,
                 stride,
                 padding,
                 activation,
                 bn,
                 init,
                 conv=None,
                 batch_norm=None,
                 bias=True,
                 preact=False,
                 name="",
                 instance_norm=False,
                 instance_norm_func=None):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(in_size,
                         out_size,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size,
                                             affine=False,
                                             track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size,
                                             affine=False,
                                             track_running_stats=False)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn",
                        batch_norm(in_size, eps=1e-6, momentum=0.99))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv1d(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal_,
                 bias: bool = True,
                 preact: bool = False,
                 name: str = "",
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=nn.Conv1d,
                         batch_norm=BatchNorm1d,
                         bias=bias,
                         preact=preact,
                         name=name,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm1d)


class Conv2d(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal_,
                 bias: bool = True,
                 preact: bool = False,
                 name: str = "",
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=nn.Conv2d,
                         batch_norm=BatchNorm2d,
                         bias=bias,
                         preact=preact,
                         name=name,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm2d)


class Conv2d_Transpose(_ConvBase):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 kernel_size: Tuple[int, int] = (1, 1),
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0),
                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                 bn: bool = False,
                 init=nn.init.kaiming_normal_,
                 bias: bool = True,
                 preact: bool = False,
                 name: str = "",
                 instance_norm=False):
        super().__init__(in_size,
                         out_size,
                         kernel_size,
                         stride,
                         padding,
                         activation,
                         bn,
                         init,
                         conv=nn.ConvTranspose2d,
                         batch_norm=BatchNorm2d,
                         bias=bias,
                         preact=preact,
                         name=name,
                         instance_norm=instance_norm,
                         instance_norm_func=nn.InstanceNorm2d)


class FC(nn.Sequential):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 *,
                 activation=nn.ReLU(inplace=True),
                 bn: bool = False,
                 init=None,
                 preact: bool = False,
                 name: str = ""):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):
    def __init__(self,
                 model,
                 bn_lambda,
                 last_epoch=-1,
                 setter=set_bn_momentum_default):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(
                type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
