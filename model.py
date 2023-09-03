import torch
from torch.nn import Linear
from ECD_CDGIConv import ECD_CDGIConv
import torch.nn as nn
import torch.nn.functional as F


class ECD_CDGINet(torch.nn.Module):
    def __init__(self, args, weights=[0.50, 0.50, 0.50, 0.50]):
        super().__init__()
        self.args = args
        in_channels = self.args.in_channels
        hidden_channels = self.args.hidden_channels
        self.linear1 = Linear(58, hidden_channels)
        self.linear_r0 = Linear(hidden_channels, 1)
        self.linear_r1 = Linear(hidden_channels, 1)
        self.linear_r2 = Linear(hidden_channels, 1)
        self.linear_r3 = Linear(hidden_channels, 1)
        # Attention weights on outputs of different convolutional layers
        self.weight_r0 = torch.nn.Parameter(torch.Tensor([weights[0]]), requires_grad=True)
        self.weight_r1 = torch.nn.Parameter(torch.Tensor([weights[1]]), requires_grad=True)
        self.weight_r2 = torch.nn.Parameter(torch.Tensor([weights[2]]), requires_grad=True)
        self.weight_r3 = torch.nn.Parameter(torch.Tensor([weights[3]]), requires_grad=True)
        self.convs1 = ECD_CDGIConv(hidden_channels, hidden_channels, kernel='simple', use_graph=True,
                                   use_weight=False)
        self.use_bn = True
        self.residual = True
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, data):
        x_input = data.x
        edge_index_1 = data.edge_index
        x_input = F.dropout(x_input, p=0.5, training=self.training)
        T0 = R0 = torch.relu(self.linear1(x_input))
        layer_ = []
        layer_.append(R0)
        i = 0
        R0 = self.convs1(R0, R0, edge_index_1, None)
        if self.residual:
            R0 = self.alpha * R0 + (1 - self.alpha) * layer_[i]  # 残差
        if self.use_bn:
            R0 = self.bns[i](R0)
        T1 = R0
        layer_ = []
        layer_.append(R0)
        R0 = self.convs1(R0, R0, edge_index_1, None)
        if self.residual:
            R0 = self.alpha * R0 + (1 - self.alpha) * layer_[i]
        if self.use_bn:
            R0 = self.bns[i](R0)
        T2 = R0
        layer_ = []
        layer_.append(R0)
        R0 = self.convs1(R0, R0, edge_index_1, None)
        if self.residual:
            R0 = self.alpha * R0 + (1 - self.alpha) * layer_[i]
        if self.use_bn:
            R0 = self.bns[i](R0)
        T3 = R0
        T0 = F.dropout(T0, p=0.5, training=self.training)
        res0 = self.linear_r0(T0)
        T1 = F.dropout(T1, p=0.5, training=self.training)
        res1 = self.linear_r1(T1)
        T2 = F.dropout(T2, p=0.5, training=self.training)
        res2 = self.linear_r2(T2)
        T3 = F.dropout(T3, p=0.5, training=self.training)
        res3 = self.linear_r3(T3)
        out = res0 * self.weight_r0 + res1 * self.weight_r1 + res2 * self.weight_r2 + res3 * self.weight_r3
        return out
