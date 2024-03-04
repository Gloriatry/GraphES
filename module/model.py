from module.layer import *
from torch import nn
from module.sync_bn import SyncBatchNorm
from helper import context as ctx


class GNNBase(nn.Module):

    def __init__(self, layer_size, activation, dropout=0.5, norm='layer', n_linear=0,
                 es=False, lam=0.1, sigma=0.5):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.n_linear = n_linear
        self.es = es
        self.lam = lam
        self.sigma = sigma

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

# class GraphSAGE(GNNBase):

#     def __init__(self, layer_size, activation, dropout=0.5, norm='layer', train_size=None, n_linear=0,
#                  es=False, lam=0.1, sigma=0.5, es_location=',0'):
#         super(GraphSAGE, self).__init__(layer_size, activation, dropout, norm, n_linear, es, lam, sigma)
#         for i in range(self.n_layers):
#             if i < self.n_layers - self.n_linear - 1:# 前n_layers - n_linear层: GraphSAGELayer层
#                 self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1]))
#                 if self.es:
#                     self.layers.append(EmbeddingSelector(layer_size[i + 1], self.sigma))
#             elif i < self.n_layers - self.n_linear:
#                 self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1]))
#             else:#后n_linear层:Linear层
#                 self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
#             if i < self.n_layers - 1 and self.use_norm:
#                 if norm == 'layer':
#                     self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
#                 elif norm == 'batch':
#                     self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
    
#     def forward(self, g, feat, in_deg=None):
#         reg_loss = 0
#         h = feat
#         mask = torch.ones(h.shape[1])  # 初始化mask，feature全传
#         for i in range(self.n_layers):
#             if self.es:
#                 if i < self.n_layers - self.n_linear - 1:
#                     if self.training:
#                         h = ctx.buffer.update(i, h, mask)
#                     h = self.dropout(h)
#                     h = self.layers[2*i](g, h, in_deg)
#                     h, stochastic_gate = self.layers[2*i+1](h)
#                     # print(stochastic_gate)
#                     # mask = self.layers[2*i+1].mu > 0
#                     # mask = stochastic_gate.bool()
#                     mask = stochastic_gate
#                     reg_loss += torch.mean(self.regularizer((self.layers[2*i+1].mu + 0.5)/self.sigma))
#                     # h = self.layers[2*i](h)
#                     # if self.training:
#                     #     mask = self.layers[2*i].mu > 0
#                     #     h = ctx.buffer.update(i, h, mask)
#                     # h = self.dropout(h)
#                     # h = self.layers[2*i+1](g, h, in_deg)
#                     # reg_loss += torch.mean(self.regularizer((self.layers[2*i].mu + 0.5)/self.sigma))
#                 elif i < self.n_layers - self.n_linear:
#                     if self.training:
#                         h = ctx.buffer.update(i, h, mask)
#                     h = self.dropout(h)
#                     h = self.layers[2*i](g, h, in_deg)
#                 else:
#                     h = self.dropout(h)
#                     h = self.layers[self.n_layers-self.n_linear-1+i](h)
#             else:
#                 if i < self.n_layers - self.n_linear:
#                     if self.training:
#                         h = ctx.buffer.update(i, h, None) # 输入的h维度是inner node，输出的维度是g中所有节点
#                     h = self.dropout(h)
#                     h = self.layers[i](g, h, in_deg) # 输入的是g中所有节点，输出是inner node
#                 else:
#                     h = self.dropout(h)
#                     h = self.layers[i](h)

#             if i < self.n_layers - 1:
#                 if self.use_norm:
#                     h = self.norm[i](h)
#                 h = self.activation(h)
#         if self.training:
#             return h, reg_loss
#         else:
#             return h
    
#     def regularizer(self, x):
#         ''' Gaussian CDF. '''
#         return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

class GraphSAGE(GNNBase):

    def __init__(self, layer_size, activation, dropout=0.5, norm='layer', train_size=None, n_linear=0,
                 es=False, lam=0.1, sigma=0.5, es_location=',0'):
        super(GraphSAGE, self).__init__(layer_size, activation, dropout, norm, n_linear, es, lam, sigma)
        self.es_location = es_location.split(',')[1:]
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:# 前n_layers - n_linear层: GraphSAGELayer层
                if self.es and str(i) in self.es_location:
                    self.layers.append(EmbeddingSelector(layer_size[i], self.sigma))
                self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1]))
            else:#后n_linear层:Linear层
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
    
    def forward(self, g, feat, in_deg=None):
        reg_loss = 0
        h = feat
        # mask = torch.ones(h.shape[1])  # 初始化mask，feature全传
        layer_idx = 0
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.es and str(i) in self.es_location:
                    h, stochastic_gate = self.layers[layer_idx](h)
                    mask = stochastic_gate
                    reg_loss += torch.mean(self.regularizer((self.layers[layer_idx].mu + 0.5)/self.sigma))
                    layer_idx += 1
                else:
                    mask = torch.ones(h.shape[1])
                if self.training:
                    h = ctx.buffer.update(i, h, mask)
                h = self.dropout(h)
                h = self.layers[layer_idx](g, h, in_deg)
                layer_idx += 1
            else:
                h = self.dropout(h)
                h = self.layers[layer_idx](h)
                layer_idx += 1

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)
        if self.training:
            return h, reg_loss
        else:
            return h
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))    