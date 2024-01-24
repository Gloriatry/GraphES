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

class GraphSAGE(GNNBase):

    def __init__(self, layer_size, activation, dropout=0.5, norm='layer', train_size=None, n_linear=0,
                 es=False, lam=0.1, sigma=0.5):
        super(GraphSAGE, self).__init__(layer_size, activation, dropout, norm, n_linear, es, lam, sigma)
        for i in range(self.n_layers):  
            if i < self.n_layers - self.n_linear:# 前n_layers - n_linear层: GraphSAGELayer层
                if self.es:
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
        # 为选择embedding引入的正则项
        reg_loss = 0
        h = feat
        for i in range(self.n_layers):
            if self.es:
                if i < self.n_layers - self.n_linear:
                    h = self.layers[2*i](h)
                    # TODO: 生成mask
                    # 思路是本地的inner node的h是直接不变，传的是经过了置0的
                    # 1.23思路是直接将h中为0的不传，依据这个生成mask
                    # TODO 这里将mask传给buffer中的一个函数，该函数生成embedding idx并resize
                    if self.training:
                        mask = self.layers[2*i].mu
                        ctx.buffer.setEmbedInfo(mask, i)
                        h = ctx.buffer.update(i, h)
                    h = self.dropout(h)
                    h = self.layers[2*i+1](g, h, in_deg)
                    reg_loss += torch.mean(self.regularizer((self.layers[2*i].mu + 0.5)/self.sigma))
                else:
                    h = self.dropout(h)
                    h = self.layers[self.n_layers-self.n_linear+i](h)
            else:
                if i < self.n_layers - self.n_linear:
                    if self.training:
                        h = ctx.buffer.update(i, h) # 输入的h维度是inner node，输出的维度是g中所有节点
                    h = self.dropout(h)
                    h = self.layers[i](g, h, in_deg) # 输入的是g中所有节点，输出是inner node
                else:
                    h = self.dropout(h)
                    h = self.layers[i](h)

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
    