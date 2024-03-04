import torch
from multiprocessing.pool import ThreadPool
from helper.timer.timer import *
from helper.Buff_unit_async_concat import *
import queue
import numpy as np
import dgl.function as fn

class Buffer(object):

    def __init__(self):
        super(Buffer, self).__init__()
        self._rank = None
        self._size = None
        self._num_in = None
        self._num_all = None
        self._boundary = []
        self._n_layers = 0
        self._layer_size = []
        self._use_async = False
        # add 2.1 - 
        self._use_sample = False
        # add 2.1 +
        self._epoch = 0
        self._buff = []
        self._embed_buf = []
        self._embed_recv, self._grad_recv = [], []
        self._recv_shape, self._send_shape = [], []
        self._pool = None
        self._step = None
        self._backend = None
        self._pl, self._pr = [], []
        self._comm_stream = []
        self.es = None

    def __init_pl_pr(self):
        self._pl, self._pr = [], []
        tot = self._num_in  # inner nodes 的个数
        for s in self._recv_shape:
            if s is None:
                self._pl.append(None)
                self._pr.append(None)
            else:
                self._pl.append(tot)
                tot += s
                self._pr.append(tot)

    # change 2.1 -
    def init_buffer(self, graph, feat, num_in, num_all, boundary, f_recv_shape, layer_size,
                    use_sample = False, sample_rate =1, sample_method = 'random',
                    use_async=False, stale_t= 1, backend='gloo', es=False):
    # change 2.1 +
        rank, size = dist.get_rank(), dist.get_world_size()
        self._rank = rank
        self._size = size
        self._num_in = num_in
        self._num_all = num_all  # 包括inner node和boundary node
        self._boundary = boundary
        self._n_layers = len(layer_size) # 不包括最后的linear层
        self._layer_size = layer_size
        self._use_async = use_async
        # add 2.1 -
        self._use_sample = use_sample
        self._sample_rate = sample_rate
        self._sample_method = sample_method
        if use_sample:
            self.sample_matrix = torch.zeros([graph.num_edges()],device='cuda') + self._sample_rate
        # add 2.1 +
        self._epoch = 0
        self._recv_shape = f_recv_shape
        self._stale_t = stale_t # indicate how many step async used
        self._step = 0 #indicate which step in
        self._backend = backend
        self.es = es
        # stream
        for i in range(stale_t):
            self._comm_stream.append(torch.cuda.Stream())
        self._send_shape = [None] * size
        for i in range(size):
            if i ==rank:
                continue
            self._send_shape[i] = boundary[i].shape[0]

        self._embed_buf = [None] * self._n_layers # 存放本地节点和remote节点连接后的embedding
        self._embed_recv, self._grad_recv = [None] * self._n_layers, [None] * self._n_layers 
        # 存放从其他worker上传输的embedding和embedding的梯度
        for i in range(self._n_layers):
            self._embed_buf[i] = torch.zeros([num_all, self._layer_size[i]], device='cuda', requires_grad=True)
            tmp1, tmp2 = [], []
            for j in range(size):
                if j == rank:
                    tmp1.append(None)
                    tmp2.append(None)
                else:
                    s1 = torch.Size([self._recv_shape[j], self._layer_size[i]])
                    s2 = torch.Size([self._send_shape[j], self._layer_size[i]])
                    tmp1.append(torch.zeros(s1, device='cuda', requires_grad=False))
                    tmp2.append(torch.zeros(s2, device='cuda', requires_grad=False))
            self._embed_recv[i] = tmp1
            # if i > 0:
            #     self._grad_recv[i] = tmp2
            self._grad_recv[i] = tmp2
        self.__init_pl_pr()  # 
        if use_async:
            for i in range(self._stale_t):
                # add 2.1 -
                self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend, use_sample=use_sample, es=self.es))
                # add 2.1 +
                self._pool = ThreadPool(processes=2*self._n_layers*self._stale_t)
        else:
            # add 2.1 -
            self._buff.append(Buff_Unit(self._send_shape, self._recv_shape, self._n_layers, self._layer_size, rank, size, backend, use_sample=use_sample, es=self.es))
            # add 2.1 +
    def nextEpoch(self):
        self._epoch += 1
        self._step = (self._step + 1) % self._stale_t

    # add 2.1 -
    def setCommuInfo(self):
        if self._use_sample:
            self._buff[self._step].setCommuInfo()       
    def initSampleNodes(self, original_graph, epoch):
        return self._buff[self._step].sampleNodes(original_graph,self.sample_matrix, self._sample_method, epoch, self._stale_t*int(self._use_async), self._pl, self._pr)

    def sampleNodes(self, original_graph, epoch):
        next_step = (self._step + 1) % self._stale_t
        return self._buff[next_step].sampleNodes(original_graph, self.sample_matrix, self._sample_method, epoch, self._stale_t*int(self._use_async), self._pl, self._pr)
    # add 2.1 + 
    
    ###############改动###############
    def __feat_fusion(self, layer, step):  # 这里的step是self._step
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                # 将embed_recv置0,相当于将embedding中没有选中的维度置0
                # change 2.1 -
                zero_tensor = torch.zeros(torch.Size([self._buff[step].get_recv_idx()[i].shape[0], self._embed_recv[layer][i].shape[1]]), device='cuda')
                self._embed_recv[layer][i][self._buff[step].get_recv_idx()[i]] = zero_tensor
                selected_recv_embed_idx = self._buff[step].get_f_recv_gpu(layer)[i][-1]
                selected_recv_embed_idx = torch.tensor(selected_recv_embed_idx, dtype=torch.int64)
                self._embed_recv[layer][i][self._buff[step].get_recv_idx()[i]][:, selected_recv_embed_idx] = self._buff[step].get_f_recv_gpu(layer)[i][:-1]
                # add 2.1 + 
                
    def __feat_concat(self, layer, feat):
        rank, size = dist.get_rank(), dist.get_world_size()
        tmp = [feat]
        for i in range(size):
            if i != rank:
                tmp.append(self._embed_recv[layer][i])
        return torch.cat(tmp)

    def update(self, layer, feat, mask):  # 这个feat代表的是h，也就是embedding
        torch.cuda.current_stream().synchronize()
        if self._use_async is False:
            with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'): # 记录前向传播的时间
                self.__feat_transfer(self._epoch, self._step, layer, feat, mask) # 传embedding
                torch.cuda.current_stream().wait_event(self._buff[self._step].get_f_cuda_event(layer)) # 确保embedding传输已完成
            self._embed_buf[layer] = self.__feat_concat(layer, feat) # 将传输的特征与当前层的特征连接起来
            if self._embed_buf[layer].requires_grad: # 
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer, self._step))
            return self._embed_buf[layer]
        else:
            if self._epoch > self._stale_t - 1: # 大于的话就可以使用tau轮前传输的embedding
                with comm_timer.timer(f'epoch:{self._epoch} forward_{layer}'):
                    self._buff[self._step].get_f_cpu_event(layer).wait()
                    torch.cuda.current_stream().wait_event(self._buff[self._step].get_f_cuda_event(layer))
                    self._buff[self._step].get_f_cpu_event(layer).clear()  # Event.wait() will block
            # 注意！！！ 2.1 有点采样时，不能在这里fusion ，不然fusion的node和接受的node将不一致
            # self.__feat_fusion(layer, step)  # 将接收到的embedding存放在self._embed_recv中, ！！！放在外面？
            # 注意！！！ 2.1
            self._embed_buf[layer] = self.__feat_concat(layer, feat)  
            # 拼接本地节点和boundary的embedding，boundary的embedding在_embed_recv中
            self._pool.apply_async(self.__feat_transfer, args=(self._epoch, self._step, layer, feat, mask))
            if self._embed_buf[layer].requires_grad and layer > 0 :
                self._embed_buf[layer].register_hook(self.__grad_hook(self._epoch, layer, self._step))
            return self._embed_buf[layer]
    

    @torch.no_grad()
    # change 2.1 -
    def __gloo_all_to_all(self, send_gpu, recv_gpu, send_cpu, recv_cpu,
                        selected_send_idx, selected_recv_idx, 
                        step, tag, layer, forward = True):
    # change 2.1 + 
        rank, size = dist.get_rank(), dist.get_world_size()
        req1, req2 = [], queue.Queue()
        self._comm_stream[step].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream[step]):
            for i in range(1, size):
                left = (rank - i + size) % size
                right = (rank + i) % size
                r2 = dist.irecv(recv_cpu[left], tag=tag, src=left)
                req2.put((r2, left))
                # prepare send tensor
                if forward:
                    # 前向传播传boundary中的send_idx部分
                    # change 2.1 - 
                    mask = send_gpu[-1]
                    if layer == 0:
                        selected_send_embed_idx = torch.nonzero(mask, as_tuple=True)[0]
                    else:
                        _, sorted_indices = torch.sort(mask, descending=True)
                        # selected_send_embed_idx = torch.nonzero(mask, as_tuple=True)[0]
                        selected_send_embed_idx = sorted_indices[:int(1*send_gpu.shape[1])]
                    tmp = [send_gpu[self._boundary[right]][selected_send_idx[right]][:, selected_send_embed_idx]]
                    selected_send_embed_idx = torch.unsqueeze(selected_send_embed_idx, 0)
                    tmp.append(selected_send_embed_idx)
                    send_cpu[right].copy_(torch.cat(tmp))
                    # change 2.1 + 
                else:
                    # 反向传播传recv_idx部分
                    # change 2.1 -
                    send_cpu[right].copy_(send_gpu[self._pl[right]:self._pr[right]][selected_recv_idx[right]]) # 这里传的是什么？
                    # change 2.1 + 
                r1 = dist.isend(send_cpu[right], tag=tag, dst=right)
                req1.append(r1)

            while not req2.empty():
                r, rank_inx = req2.get()
                # TODO: if r.is_completed() run following lines else next r (see issue #30723)
                r.wait()
                if forward:
                    recv_gpu[rank_inx].copy_(recv_cpu[rank_inx], non_blocking = True)
                else:
                    recv_gpu[rank_inx].copy_(recv_cpu[rank_inx], non_blocking = True)
            # TODO: remove this 'wait'
            for r in req1:
                r.wait()

    ###############改动###############
    def __feat_transfer(self, epoch, step, layer, feat, mask):
        if self.es:
            self._buff[step].setEmbedInfo(mask, layer)
        tag = TransferTag.FEAT_BEGIN + epoch * 2 * self._n_layers + layer
        tmp = [feat]
        mask = torch.unsqueeze(mask, 0)
        tmp.append(mask.to('cuda:0'))
        if self._backend == 'gloo':
            with torch.no_grad():
                # send_gpu, recv_gpu, send_cpu, recv_cpu
                # change 2.1 -
                self.__gloo_all_to_all(torch.cat(tmp), self._buff[step].get_f_recv_gpu(layer), self._buff[step].get_f_send_cpu(layer), self._buff[step].get_f_recv_cpu(layer),
                                    self._buff[step].get_send_idx(), self._buff[step].get_recv_idx(), 
                                    step, tag, layer, forward=True)
                # change 2.1 + 
            self._buff[step].get_f_cuda_event(layer).record(self._comm_stream[step]) # 在通信流上记录CUDA事件，以便后续等待
        else:
            raise NotImplementedError
        # 注意！！！ 2.1 ，会导致上次说的一个问题！，即
        self.__feat_fusion(layer, step)  # 将接收到的embedding存放在self._embed_recv中, ！！！
        # 注意！！！ 2.1
        self._buff[step].get_f_cpu_event(layer).set() # Event.wait() will not block
        #add at 7.24
        self._buff[step].update_f_event[layer].set()


    def __grad_fusion(self, layer, step):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i != rank:
                # change 2.1 -
                self._grad_recv[layer][i][self._buff[step].get_send_idx()[i]] = self._buff[step].get_g_recv_gpu(layer)[i]
                # change 2.1 + 

    def __update_grad(self, layer, grad):
        rank, size = dist.get_rank(), dist.get_world_size()
        for i in range(size):
            if i == rank:
                continue
            else:
                grad[self._boundary[i]] += self._grad_recv[layer][i]

    def __grad_hook(self, epoch, layer, step):
        def fn(grad):
            torch.cuda.current_stream().synchronize()
            if self._use_async is False:
                with comm_timer.timer(f'epoch:{epoch} backward_{layer}'):
                    self.__grad_transfer(epoch, step, layer, grad)
                    torch.cuda.current_stream().wait_event(self._buff[step].get_b_cuda_event(layer))
                self.__update_grad(layer, grad)
                return grad
            else:
                if self._epoch > self._stale_t - 1:
                    with comm_timer.timer(f'epoch:{epoch} backward_{layer}'):
                        self._buff[step].get_b_cpu_event(layer).wait()   # wait last eopch backward finish : self._b_cpu_event[layer].set()
                        torch.cuda.current_stream().wait_event(self._buff[step].get_b_cuda_event(layer))
                        self._buff[step].get_b_cpu_event(layer).clear()  # Event.wait() will not block
                self.__update_grad(layer, grad)
                self._pool.apply_async(self.__grad_transfer, args=(epoch, step, layer, grad), error_callback=lambda x:print('error!!!'))
                return grad
        return fn
    
    def __grad_transfer(self, epoch, step, layer, grad):
        tag = TransferTag.FEAT_BEGIN + epoch * 2 * self._n_layers + layer + self._n_layers
        if self._backend == 'gloo':
            # change 2.1 - 
            self.__gloo_all_to_all(grad, self._buff[step].get_g_recv_gpu(layer), self._buff[step].get_g_send_cpu(layer), self._buff[step].get_g_recv_cpu(layer),
                                   self._buff[step].get_send_idx(), self._buff[step].get_recv_idx(), step, tag, layer, forward=False)
            self._buff[step].get_b_cuda_event(layer).record(self._comm_stream[step])
            # change 2.1 + 
        else:
            raise NotImplementedError
        self.__grad_fusion(layer, step)
        self._buff[step].get_b_cpu_event(layer).set()  # Event.wait() will not block
        #add at 7.24
        self._buff[step].update_b_event[layer].set()
    
    def wait(self):
        for i in range(self._n_layers):
            self._buff[self._step].update_f_event[i].wait()
            self._buff[self._step].update_f_event[i].clear()
        for i in range(1,self._n_layers):
            self._buff[self._step].update_b_event[i].wait()
            self._buff[self._step].update_b_event[i].clear()
        #add at 7.24