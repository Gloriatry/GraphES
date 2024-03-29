import torch
import torch.distributed as dist
from multiprocessing import Event
import dgl
import time
from helper.transfer_tag import *


class Buff_Unit(object):
    def __init__(self, send_num_tot, recv_num_tot, n_layers, layer_size, rank, size, backend, use_sample,
                es=False) -> None:
        super(Buff_Unit, self).__init__()
        self._n_layers = n_layers
        self._layer_size = layer_size
        self._rank = rank
        self._size = size
        self.es = es
        self._backend = backend
        self._send_num_tot = send_num_tot
        self._recv_num_tot = recv_num_tot
        # cpu f/g send/recv
        self.f_send_cpu, self.g_send_cpu = [], []
        self.f_recv_cpu, self.g_recv_cpu = [], []
        self.f_send_cpu, self.g_send_cpu = [None] * n_layers, [None] * n_layers
        self.f_recv_cpu, self.g_recv_cpu = [None] * n_layers, [None] * n_layers
        # gpu f/g recv
        self.f_recv_gpu, self.g_recv_gpu = [], []
        self.f_recv_gpu, self.g_recv_gpu = [None] * n_layers, [None] * n_layers
        # event

        # add at 7.24 by wsl
        self.update_f_event = [None] * n_layers
        self.update_b_event = [None] * n_layers
        for i in range(n_layers):
            self.update_f_event[i] = Event()
            self.update_b_event[i] = Event()
            self.update_f_event[i].set()
            self.update_b_event[i].set()
        # add at 7.24 by wsl
        self.f_cpu_event, self.b_cpu_event = [None] * n_layers, [None] * n_layers
        self.f_cuda_event, self.b_cuda_event = [None] * n_layers, [None] * n_layers
        for i in range(n_layers):
            self.f_cpu_event[i] = Event()
            self.b_cpu_event[i] = Event()
            self.f_cuda_event[i] = torch.cuda.Event()
            self.b_cuda_event[i] = torch.cuda.Event()

        # add 2.1 -
        # select nodes index
        self.send_idx, self.recv_idx = [], []
        self.send_idx, self.recv_idx = [None] * size, [None] * size
        # add 2.1 +

        # embedding index
        self.send_embed_idx, self.recv_embed_idx = [], []
        self.send_embed_idx, self.recv_embed_idx = [None] * n_layers, [None] * n_layers

        # add 2.1 -
        self.send_num, self.recv_num = [None] * size, [None] * size
        self.send_idx_temp, self.recv_idx_temp = [None] * size, [None] * size
        # add 2.1 +

        self.__initBuffUnit(send_num_tot, recv_num_tot, n_layers, layer_size, rank, size, backend)
        # add 2.1 -
        self.__initSenRecvIdx(send_num_tot, recv_num_tot, rank, size)
        # add 2.1 +
        self.__initEmbedIdx(n_layers, layer_size, rank, size)

    ###############改动###############
    def __initEmbedIdx(self, n_layers, layer_size, rank, size):
        for i in range(n_layers):
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    tmp1.append(torch.arange(layer_size[i], dtype=torch.long, device='cuda'))
                    tmp2.append(torch.arange(layer_size[i], dtype=torch.long, device='cuda'))
            self.send_embed_idx[i] = tmp1
            self.recv_embed_idx[i] = tmp2

    def __initBuffUnit(self, send_num_tot, recv_num_tot, n_layers, layer_size, rank, size, backend):
        # f/g  send/recv in cpu memory
        if backend == 'gloo':
            for i in range(n_layers):
                tmp1, tmp2, tmp3, tmp4 = [], [], [], []
                for j in range(size):
                    if j == rank:
                        tmp1.append(None)
                        tmp2.append(None)
                        tmp3.append(None)
                        tmp4.append(None)

                    else:
                        s1 = torch.Size([send_num_tot[j], layer_size[i]])
                        s2 = torch.Size([recv_num_tot[j], layer_size[i]])
                        tmp1.append(torch.zeros(s1)) #,pin_memory=True
                        tmp2.append(torch.zeros(s2))
                        tmp3.append(torch.zeros(s2))
                        tmp4.append(torch.zeros(s1))
                self.f_send_cpu[i] = tmp1
                self.f_recv_cpu[i] = tmp3
                if i > 0:
                    self.g_send_cpu[i] = tmp2
                    self.g_recv_cpu[i] = tmp4
                # self.g_send_cpu[i] = tmp2
                # self.g_recv_cpu[i] = tmp4
        # f/g recv in gpu
        for i in range(n_layers):
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    s1 = torch.Size([recv_num_tot[j], layer_size[i]])
                    s2 = torch.Size([send_num_tot[j], layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda'))
                    tmp2.append(torch.zeros(s2, device='cuda'))
            self.f_recv_gpu[i] = tmp1
            if i > 0:
                self.g_recv_gpu[i] = tmp2
            # self.g_recv_gpu[i] = tmp2

    ###############改动###############
    def setEmbedInfo(self, mask, layer):
        rank, size = dist.get_rank(), dist.get_world_size()
        recv_embed_idx_cpu = [None] * size
        send_embed_idx_cpu = [None] * size
        num_embed_send = [None] * size

        for i in range(size):
            if i == rank:
                continue
            num_embed_send[i] = mask.bool().sum().to('cpu')
            # if i == (rank + 1):
            #     print(num_embed_send[i]/self._layer_size[layer])
            self.send_embed_idx[layer][i] = torch.nonzero(mask, as_tuple=True)[0]  ### mask是send_embed_idx还在recv_embed_idx???
            send_embed_idx_cpu[i] = self.send_embed_idx[layer][i].to('cpu')

        # 进程之间互相传embed index
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            num_embed_recv = torch.tensor([0])
            req = dist.isend(num_embed_send[right], dst=right, tag=TransferTag.EMBED1)
            dist.recv(num_embed_recv, src=left, tag=TransferTag.EMBED1)
            recv_embed_idx_cpu[left] = torch.zeros(num_embed_recv, dtype=torch.long)
            self.recv_embed_idx[layer][left] = torch.zeros(num_embed_recv, dtype=torch.long, device='cuda')
            req.wait()
            req = dist.isend(send_embed_idx_cpu[right], dst=right, tag=TransferTag.EMBED2)
            dist.recv(recv_embed_idx_cpu[left], src=left, tag=TransferTag.EMBED2)
            req.wait()
        
        for i in range(size):
            if i == rank:
                continue
            self.recv_embed_idx[layer][i].copy_(recv_embed_idx_cpu[i])
        
        num_embed_recv = [None] * size
        for i in range(size):
            if i == rank:
                continue
            else:
                num_embed_recv[i] = self.recv_embed_idx[layer][i].shape[0]
        
        # resize发送和接收变量
        for i in range(size):
            if i == rank:
                continue
            else:
                node_num = self.f_send_cpu[layer][i].shape[0]
                self.f_send_cpu[layer][i].resize_([node_num, num_embed_send[i]])
                node_num = self.f_recv_cpu[layer][i].shape[0]
                self.f_recv_cpu[layer][i].resize_([node_num, num_embed_recv[i]])
                node_num = self.f_recv_gpu[layer][i].shape[0]
                self.f_recv_gpu[layer][i].resize_([node_num, num_embed_recv[i]])


    # communicating idx Tag
    # add 2.1 -
    def get_send_idx(self):
        return self.send_idx
    def get_recv_idx(self):
        return self.recv_idx
    # add 2.1 +
    def get_send_embed_idx(self, layer):
        return self.send_embed_idx[layer]
    def get_recv_embed_idx(self, layer):
        return self.recv_embed_idx[layer]
    
    def get_f_send_cpu(self, layer):
        return self.f_send_cpu[layer]
    def get_g_send_cpu(self, layer):
        return self.g_send_cpu[layer]
    def get_f_recv_cpu(self, layer):
        return self.f_recv_cpu[layer]
    def get_g_recv_cpu(self, layer):
        return self.g_recv_cpu[layer]
    def get_f_recv_gpu(self, layer):
        return self.f_recv_gpu[layer]
    def get_g_recv_gpu(self, layer):
        return self.g_recv_gpu[layer]

    def get_f_cpu_event(self, layer):
        return self.f_cpu_event[layer]
    def get_b_cpu_event(self, layer):
        return self.b_cpu_event[layer]
    def get_f_cuda_event(self, layer):
        return self.f_cuda_event[layer]
    def get_b_cuda_event(self, layer):
        return self.b_cuda_event[layer]
    

    # add 2.1 -
    def __initSenRecvIdx(self, send_num_tot, recv_num_tot, rank, size):
        """
        If don't use sample, then the send/recv index should be initialized
        """
        for i in range(size):
            if i == rank:
                continue
            self.send_idx[i] = torch.arange(send_num_tot[i], dtype=torch.long, device='cuda')
            self.recv_idx[i] = torch.arange(recv_num_tot[i], dtype=torch.long, device='cuda')
    def sampleNodes(self, original_graph, sample_matrix, sample_method, epoch, stale_t, pl, pr):
        """
        when the process communicate with other processes, select partial neighbor nodes can efficiently reduce data volume, so then reduce
        communication time.
        there are three methods to select neighbor nodes, which represent the nodes in other partition but there is an edge pointing towards the nodes in this partition
        """
        return self.__randomSample(original_graph, sample_matrix, sample_method, epoch, stale_t, pl, pr)
    def __randomSample(self, graph, sample_matrix, sample_method, epoch, stale_t, pl, pr):
        rank, size = dist.get_rank(), dist.get_world_size()
        
        # sample
        sample_mask = torch.bernoulli(sample_matrix).bool()  # 二项分布，p为sample_matrix中对应值

        # relabel_nodes=False,will not remove the isolated nodes and won't relabel the incident nodes in the extracted subgraph.
        g = dgl.edge_subgraph(graph, sample_mask, relabel_nodes=False, store_ids=False, output_device='cuda')
        # g.add_edges(u=g.nodes('_V'),v=g.nodes('_V'),etype='_E')
        mask = g.has_edges_between(u=g.nodes('_V'),v=g.nodes('_V'),etype='_E')
        add_edge_idx = torch.nonzero(mask == False, as_tuple=True)[0].to(torch.int32)
        g.add_edges(u=add_edge_idx,v=add_edge_idx,etype='_E')

        # prepare idx data
        recv_idx_cpu = [None] * size
        send_idx_cpu = [None] * size
        num_recv = [None] * size

        g_outdeg = g.out_degrees(u='__ALL__', etype='_E')  # 返回对应节点的出度
        for i in range(size):
            if i == rank:
                continue
            temp = g_outdeg[pl[i]:pr[i]]
            num_recv[i] = temp.bool().sum().to('cpu')  # 该进程采样进程i中的节点个数
            self.recv_idx_temp[i] = torch.nonzero(temp, as_tuple=True)[0]  # 出度不为0的节点，也就是被采样的节点
            recv_idx_cpu[i] = self.recv_idx_temp[i].to('cpu')

        # communicate for training
        for i in range(1, size):
            left = (rank - i + size) % size
            right = (rank + i) % size
            num_send = torch.tensor([0])
            req = dist.isend(num_recv[right], dst=right, tag=TransferTag.SAMPLE1)
            dist.recv(num_send, src=left, tag=TransferTag.SAMPLE1)
            send_idx_cpu[left] = torch.zeros(num_send, dtype=torch.long)
            self.send_idx_temp[left] = torch.zeros(num_send, dtype=torch.long, device='cuda')
            req.wait()
            req = dist.isend(recv_idx_cpu[right], dst=right, tag=TransferTag.SAMPLE2)
            dist.recv(send_idx_cpu[left], src=left, tag=TransferTag.SAMPLE2)
            req.wait()
            
        for i in range(size):
            if i == rank:
                continue
            self.send_idx_temp[i].copy_(send_idx_cpu[i])
        for i in range(size):
            if i ==rank:
                continue
            self.send_num[i] = self.send_idx_temp[i].shape[0]
            self.recv_num[i] = self.recv_idx_temp[i].shape[0]
        return g, g.in_degrees(v='__ALL__',etype='_E')
    def setCommuInfo(self):
        for i in range(self._size):
            if i ==self._rank:
                continue
            self.send_idx[i].resize_(self.send_idx_temp[i].shape)
            self.recv_idx[i].resize_(self.recv_idx_temp[i].shape)
            self.send_idx[i].copy_(self.send_idx_temp[i])
            self.recv_idx[i].copy_(self.recv_idx_temp[i])
        self.__resizeBuffUnit(self.send_num, self.recv_num, self._n_layers, self._layer_size, self._rank, self._size)
    def __resizeBuffUnit(self, send_num, recv_num, n_layers, layer_size, rank, size):
        for i in range(n_layers):
            for j in range(size):
                if j == rank:
                    continue
                self.f_send_cpu[i][j].resize_([send_num[j],layer_size[i]])
                self.f_recv_cpu[i][j].resize_([recv_num[j],layer_size[i]])
                self.f_recv_gpu[i][j].resize_([recv_num[j],layer_size[i]])
                if i > 0:
                    self.g_send_cpu[i][j].resize_([recv_num[j],layer_size[i]])
                    self.g_recv_cpu[i][j].resize_([send_num[j],layer_size[i]])
                    self.g_recv_gpu[i][j].resize_([send_num[j],layer_size[i]])
                # self.g_send_cpu[i][j].resize_([recv_num[j],layer_size[i]])
                # self.g_recv_cpu[i][j].resize_([send_num[j],layer_size[i]])
                # self.g_recv_gpu[i][j].resize_([send_num[j],layer_size[i]])
    # add 2.1 +