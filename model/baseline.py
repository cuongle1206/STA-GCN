import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

'''Spatial GCN module'''
class GCN_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN_module, self).__init__()
        self.linear_trans   = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A_norm=None):
        h                   = self.linear_trans(x)
        output              = torch.einsum('uv,nctv->nctu',A_norm, h)
        return output

'''Spatial modeling block (with residual connection)'''
class SA_block(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm, residual=True):
        super(SA_block, self).__init__()

        self.PA_norm        = nn.Parameter(torch.from_numpy(A_norm.astype(np.float32)))
        # self.PA_norm        = Variable(torch.from_numpy(A_norm.astype(np.float32)), requires_grad=False)
        self.gcn_calc       = GCN_module(in_channels, out_channels)
        self.bn             = nn.BatchNorm2d(out_channels)
        self.alpha          = nn.Parameter(torch.zeros(1))
        self.relu           = nn.ReLU(inplace=True)

        if not residual:
            self.residual   = lambda x: 0
        else:
            if in_channels == out_channels:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):

        A_norm              = self.PA_norm
        # A_norm              = self.PA_norm.cuda(x.get_device())
        y_gcn               = self.gcn_calc(x, A_norm)
        y                   = self.bn(y_gcn)
        y                   += self.residual(x)
        output              = self.relu(y)
        return output

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                    stride=(stride, 1), dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                        padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, residual=True,
                 residual_kernel_size=1):
        super().__init__()

        # Multiple branches of temporal convolution
        self.num_branches = 4
        branch_channels = out_channels // self.num_branches
        
        # Temporal Convolution branches
        self.branches = nn.ModuleList([])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=1),
            nn.ReLU(inplace=True),
            TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride)
        ))
        
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride)
        ))

        self.branches.append(nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out

class STA_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, A_norm, adaptive=True, stride=1, kernel_size=5, dilations=[1,2], residual=True):
        super(STA_GCN, self).__init__()
        self.spatial_att    = SA_block(in_channels, out_channels, A_norm)
        self.ms_conv        = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, residual=False)
        self.beta           = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y                   = self.spatial_att(x)
        y                   = self.ms_conv(y)
        output              = self.relu(y + self.residual(x))
        return output

class Embeddeding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Embeddeding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

# Model
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        
        A_norm              = self.graph.A_norm
        self.data_bn        = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channels       = 64
        self.embedder       = Embeddeding(in_channels, base_channels)

        self.layer1         = STA_GCN(  base_channels,   base_channels, A_norm, residual=False)
        self.layer2         = STA_GCN(  base_channels,   base_channels, A_norm)
        self.layer3         = STA_GCN(  base_channels,   base_channels, A_norm)
        self.layer4         = STA_GCN(  base_channels,   base_channels, A_norm)
        self.layer5         = STA_GCN(  base_channels, 2*base_channels, A_norm, stride=2)
        self.layer6         = STA_GCN(2*base_channels, 2*base_channels, A_norm)
        self.layer7         = STA_GCN(2*base_channels, 2*base_channels, A_norm)
        self.layer8         = STA_GCN(2*base_channels, 4*base_channels, A_norm, stride=2)
        self.layer9         = STA_GCN(4*base_channels, 4*base_channels, A_norm)
        self.layer10        = STA_GCN(4*base_channels, 4*base_channels, A_norm)
        self.fc             = nn.Linear(4*base_channels, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        if drop_out:
            self.drop_out   = nn.Dropout(drop_out)
        else:
            self.drop_out   = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC        = x.shape
            x               = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        
        N, C, T, V, M       = x.size()
        x                   = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x                   = self.data_bn(x)
        x                   = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x                   = self.embedder(x)

        x                   = self.layer1(x)
        x                   = self.layer2(x)
        x                   = self.layer3(x)
        x                   = self.layer4(x)
        x                   = self.layer5(x)
        x                   = self.layer6(x)
        x                   = self.layer7(x)
        x                   = self.layer8(x)
        x                   = self.layer9(x)
        x                   = self.layer10(x)

        # [N*M, T, C]
        c_new               = x.size(1)
        x                   = x.view(N, M, c_new, -1)
        x                   = x.mean(3)                 # Global Average Pooling (Spatial+Temporal)
        x                   = x.mean(1)                 # Average pool number of bodies in the sequence
        x                   = self.drop_out(x)
        return self.fc(x)


if __name__ == "__main__":
    # For debugging purposes
    import sys
    sys.path.append('..')
    model = Model(num_class=60, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph')
    N, C, T, V, M = 10, 3, 64, 25, 2
    x = torch.randn(N,C,T,V,M)
    model.forward(x)
    print('Model total # params:', count_params(model))
