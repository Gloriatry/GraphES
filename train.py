import argparse
import torch
import os
import torch.distributed as dist
import time
from helper.utils import *
from multiprocessing.pool import ThreadPool
import copy
import queue
from tensorboardX import SummaryWriter
import subprocess

# %%   
def main(args):
    print("start!\n")
    t_recor_start = time.time()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    # 关闭异常检测和性能分析信息
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    g, node_dict, gpb = load_partition(part_config=args.part_config, part_id=rank, inductive=args.inductive, dataset=args.dataset)
    if rank == 0 and args.eval:
        full_g, n_feat, n_class = load_data(args.dataset)
        if args.inductive:
            _, val_g, test_g = inductive_split(full_g)
        else:
            val_g, test_g = full_g.clone(), full_g.clone()
        del full_g
    if rank == 0:
        os.makedirs('checkpoint/', exist_ok=True)
        os.makedirs('results/', exist_ok=True)
    part = create_inner_graph(g.clone(), node_dict)  #创建inner graph ->part
    num_in = node_dict['inner_node'].bool().sum().item()
    part.ndata.clear()
    part.edata.clear()
    print(f'Process {rank} has {g.num_nodes()} nodes, {g.num_edges()} edges '
          f'{part.num_nodes()} inner nodes, and {part.num_edges()} inner edges.')
    g, part, node_dict = move_to_cuda(g, part, node_dict)
    with open(args.part_config, 'r') as conf_f:
        part_metadata = json.load(conf_f)
        n_feat = part_metadata['n_feat']
        n_class = part_metadata['n_class']
        n_train = part_metadata['n_train']  # n_train是什么意思？
    boundary = get_boundary(node_dict, gpb) # boundary存放的是对于本地是inner node，但是对于别人是neighbor的节点
    layer_size = get_layer_size(n_feat, n_class, args.n_hidden, args.n_layers)
    pos = get_pos(node_dict, gpb) # 得到节点相对位置的信息
    g = order_graph(part, g, gpb, node_dict, pos) # 构建包含边界节点的图，也就是每次前向传播时的完整的图
    in_deg = node_dict['in_degree'] 
    g, node_dict, boundary = move_train_first(g, node_dict, boundary) # 将训练节点移动到图的前部
    recv_shape = get_recv_shape(node_dict) # 获取接收数据的形状，是对应worker的可能传输节点的最大值
    if args.dataset == 'yelp':
        labels = node_dict['label'][node_dict['train_mask']].float()
    else:
        labels = node_dict['label'][node_dict['train_mask']]
    train_mask = node_dict['train_mask']
    part_train = train_mask.int().sum().item()
    feat = node_dict['feat'] # inner node的feat

    # change 2.1 -
    ctx.buffer.init_buffer(g, feat, num_in, g.num_nodes('_U'), boundary, recv_shape, layer_size[:args.n_layers-args.n_linear], 
                        use_sample = args.use_sample,sample_rate=args.sample_rate, sample_method = args.sample_method,
                        use_async=args.use_async, stale_t=args.async_step, backend=args.backend, es=args.use_es)  
    # change 2.1 +

    torch.manual_seed(args.seed) # necessary! Otherwise, the model parameters initialized on different GPUs will be different
    model = create_model(layer_size, n_train, args)
    model.cuda()
    ctx.reducer.init(model)
    for i, (name, param) in enumerate(model.named_parameters()):
        param.register_hook(reduce_hook(param, name, n_train))
    best_model, best_acc = None, 0
    result_file_name = f'results/{args.dataset}_{args.use_es}_{args.sigma}_{args.use_sample}_{args.sample_rate}_{args.use_async}_{args.async_step}.txt'
    if args.dataset == 'yelp':
        loss_fcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_fcn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dur, comm_dur, reduce_dur = [], [], []
    commu_time = 0
    torch.cuda.reset_peak_memory_stats()
    eval_thread = None
    eval_pool = ThreadPool(processes=1)

    # add 2.1 -
    sample_thread = None
    sample_pool = ThreadPool(processes=1)
    # add 2.1 +

    tg = g

    # add 2.1 -
    if args.use_sample:
        Q = queue.Queue()
        sample_thread = sample_pool.apply_async(ctx.buffer.initSampleNodes,args=(g,-1-args.async_step*int(args.use_async)),error_callback=lambda x:print('Init sample error!!!'))
    if args.use_async and args.use_sample:
        for i in range(args.async_step):
            sample_thread.wait()
            sg, in_deg = sample_thread.get()
            Q.put((sg, in_deg))
            sample_thread = sample_pool.apply_async(ctx.buffer.initSampleNodes,args=(g,i-args.async_step*int(args.use_async)),error_callback=lambda x:print('Init sample error!!!'))
    # add 2.1 +     

    if rank == 0:
        writer = SummaryWriter(f'logs/{args.dataset}_{args.use_es}_{args.sigma}_{args.lam}_{args.use_sample}_{args.sample_rate}_{args.use_async}_{args.async_step}_logs')
    command = f"ifconfig enp5s0f1"

    del node_dict
    loss_rc = []
    test_accuray_rc= []
    # if result_file_name is not None :
    #     with open(result_file_name, 'a+') as f:
    #         f.write(str(args) + '\n')
# %% training
    if not args.use_async:
        recv_comm, send_comm = [], []

    for epoch in range(args.n_epochs): 
        if not args.use_async:
            result_before = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]
        print(f"Epoch:{epoch}")
        t0 = time.time()
        # add at 7.24 在这个地方可以异步以model异步单位
        if args.use_async == True:
            tc = time.time()
            ctx.buffer.wait()
            commu_time = time.time() - tc  # 传输embedding的时间，为什么是这样的？
        
        # add 2.1 -
        if sample_thread is not None:
            sample_thread.wait()
            sg, in_deg = sample_thread.get()
            ctx.buffer.setCommuInfo()  
            Q.put((sg, in_deg))
            if epoch <= args.n_epochs - args.async_step:
                sample_thread = sample_pool.apply_async(ctx.buffer.sampleNodes,args=(g,epoch,),error_callback=lambda x:print('sampleNodes error!!!'))
            tg, in_deg = Q.get()
        # add 2.1 +
            
        model.train()
        if args.model == 'graphsage':
            logits, reg_loss = model(tg, feat, in_deg)
        else:
            raise Exception
        if args.inductive:
            loss = loss_fcn(logits, labels)
        else:
            loss = loss_fcn(logits[train_mask], labels)
        loss += args.lam * reg_loss  # args.use_es为False时,reg_loss为0,不影响
        del logits
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_reduce = time.time()
        ctx.reducer.synchronize()
        reduce_time = time.time() - pre_reduce
        optimizer.step()
        ctx.buffer.nextEpoch()
        train_dur.append(time.time() - t0)
        comm_dur.append(ctx.comm_timer.tot_time()+commu_time)
        reduce_dur.append(reduce_time)
        if not args.use_async:
            # time.sleep(1)
            result_after = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]
            rx_before = int(result_before.decode().split('bytes')[1].split('(')[0])
            tx_before = int(result_before.decode().split('bytes')[2].split('(')[0])
            rx_after = int(result_after.decode().split('bytes')[1].split('(')[0])
            tx_after = int(result_after.decode().split('bytes')[2].split('(')[0])
            rx_diff = rx_after - rx_before
            tx_diff = tx_after - tx_before
            recv_comm.append(rx_diff)
            send_comm.append(tx_diff)
        if (epoch + 1) % 10 == 0:
            print("Process {:03d} | Epoch {:05d} | Time(s) {:.4f} | Comm(s) {:.4f} | Reduce(s) {:.4f} | Loss {:.4f}".format(
                  rank, epoch, np.average(train_dur[-10:]), np.average(comm_dur[-10:]), np.average(reduce_dur[-10:]), loss.item() / part_train) )
            loss_rc += [round(loss.item() / part_train,4)]
        ctx.comm_timer.clear()
        del loss
        
        if rank == 0 and args.eval and (epoch + 1) % args.log_every == 0:
            model_copy = copy.deepcopy(model)
            if not args.inductive:
                eval_thread = eval_pool.apply_async(evaluate_trans, args=(args, 'Epoch %05d' % epoch, model_copy, val_g, test_accuray_rc, time.time()-t_recor_start, writer, epoch, result_file_name))
            else:
                eval_thread = eval_pool.apply_async(evaluate_induc, args=(args, 'Epoch %05d' % epoch, model_copy, val_g, 'val', test_accuray_rc, time.time()-t_recor_start, writer, epoch, result_file_name))
    if eval_thread:
        eval_thread.get()
    reocord_time(args, train_dur, comm_dur, reduce_dur, loss_rc, test_accuray_rc)
    if not args.use_async:
        print("recv_mean:", int(sum(recv_comm)/len(recv_comm)), "send_mean:", int(sum(send_comm)/len(send_comm)))
    # if args.inductive:
    #     evaluate_induc('Final Test Result', model, test_g, 'test')
    print("finish!\n")
# %% __main__  parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch distributed test")
    # distributed
    parser.add_argument("--backend",type=str, default='gloo', help="enter the backend")
    parser.add_argument("--rank",type=int, default=0, help="rank of this process")
    parser.add_argument("--world_size","--world-size", type=int, default=4, help="world size of the group")
    parser.add_argument("--master_addr",type=str, default='192.168.124.102', help="enter the backend")
    parser.add_argument("--master_port",type=str, default='6668', help="enter the backend")
    parser.add_argument("--nic_name",type=str, default='', help="the nic name to communicate")
    parser.add_argument("--device",type=int, default=0, help="training device")
    #dataset
    parser.add_argument("--dataset", type=str, default='reddit', help="the input dataset")
    #partition & location
    parser.add_argument("--graph_dir", type=str, default='', help="the input dataset")
    parser.add_argument("--part_config", type=str, default='', help="part_config - .json file")
    #model
    parser.add_argument("--n-hidden", "--n_hidden", type=int, default=16, help="the number of hidden units")
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2, help="the number of GNN layers")
    parser.add_argument("--n-linear", "--n_linear", type=int, default=0, help="the number of linear layers")
    parser.add_argument("--model", type=str, default='graphsage', help="model for training")
    # training
    parser.add_argument("--n-epochs", "--n_epochs",type=int, default=200,help="epochs of training")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0, help="weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--inductive", action='store_true', help="inductive learning setting")
    parser.add_argument('--eval', action='store_true', help="enable evaluation")
    parser.add_argument("--norm", choices=['layer', 'batch'], default='layer', help="normalization method")
    parser.add_argument("--log-every", "--log_every", type=int, default=10)
    parser.add_argument("--fix-seed", "--fix_seed", action='store_true', help="fix random seed")  # TODO set seed: torch.manual_seed(args.seed)
    parser.add_argument("--seed", type=int, default=1024)
    # async
    parser.add_argument("--use-async", "--use_async", action='store_true', help="Whether to use asynchronous transmission")
    parser.add_argument("--async-step", "--async_step", type=int, default=1, help="Number of rounds of asynchronous information used")
    # sample
    parser.add_argument("--use-sample", "--use_sample", action='store_true', help="Whether to use sample ")
    parser.add_argument("--sample-rate", "--sample_rate", type=float, default=0.1 , help="Sample rate")
    parser.add_argument("--sample-method", "--sample_method", choices=['random', ''], default='random', help="neighbor nodes' sample method cross partitions")
    # embedding sample
    parser.add_argument("--use-es", "--use_es", action='store_true', help="Whether to use embdding sampling")
    parser.add_argument("--lam", type=float, default=0.1, help="The weighting factor for the regularization")
    parser.add_argument("--sigma", type=float, default=0.5, help="Standard deviation in the Gaussian distribution")
    args = parser.parse_args()
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = args.master_addr #rank = 0 的process所在的machine的ip
    os.environ['MASTER_PORT'] = args.master_port #指定一个rank = 0 的process所在的machine的空闲port
    os.environ['GLOO_SOCKET_IFNAME'] = getNicName(args)  #necessary manual add,指出对应的网卡
    os.environ['RANK'] = str(args.rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # TODO assert parameters
    # assertParameters(args)
    dist.init_process_group(backend=args.backend, rank=args.rank, world_size=args.world_size)
    main(args)