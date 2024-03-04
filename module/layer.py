import torch
from torch import nn
import math
import dgl.function as fn


class GraphSAGELayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True):
        super(GraphSAGELayer, self).__init__()

        self.linear1 = nn.Linear(in_feats, out_feats, bias=bias)
        self.linear2 = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear1.weight.size(1))
        self.linear1.weight.data.uniform_(-stdv, stdv)
        self.linear2.weight.data.uniform_(-stdv, stdv)
        if self.linear1.bias is not None:
            self.linear1.bias.data.uniform_(-stdv, stdv)
            self.linear2.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_deg):
        with graph.local_scope():
            if self.training:
                degs = in_deg.unsqueeze(1)
                num_dst = graph.num_nodes('_V')
                graph.nodes['_U'].data['h'] = feat
                graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                        fn.sum(msg='m', out='h'),
                                        etype='_E')
                ah = graph.nodes['_V'].data['h'] / degs
                feat = self.linear1(feat[0:num_dst]) + self.linear2(ah)
            else:
                assert in_deg is None
                degs = graph.in_degrees().unsqueeze(1)
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                ah = graph.ndata.pop('h') / degs

                feat = self.linear1(feat) + self.linear2(ah)
        return feat

class GCNLayer(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 use_pp=False):
        super(GCNLayer, self).__init__()
        self.use_pp = use_pp
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, feat, in_norm, out_norm):
        with graph.local_scope():
            if self.training:
                if self.use_pp:
                    feat = self.linear(feat)
                else:
                    in_norm = in_norm.unsqueeze(1)
                    out_norm = out_norm.unsqueeze(1)
                    graph.nodes['_U'].data['h'] = feat / out_norm
                    graph['_E'].update_all(fn.copy_u(u='h', out='m'),
                                           fn.sum(msg='m', out='h'),
                                           etype='_E')
                    feat = self.linear(graph.nodes['_V'].data['h'] / in_norm)
            else:
                in_norm = torch.sqrt(graph.in_degrees()).unsqueeze(1)
                out_norm = torch.sqrt(graph.out_degrees()).unsqueeze(1)
                graph.ndata['h'] = feat / out_norm
                graph.update_all(fn.copy_u(u='h', out='m'),
                                 fn.sum(msg='m', out='h'))
                feat = self.linear(graph.ndata.pop('h') / in_norm)
        return feat

class EmbeddingSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(EmbeddingSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        # self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x, stochastic_gate
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0) 

    def _apply(self, fn):
        super(EmbeddingSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self