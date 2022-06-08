from my_model import Retriever, Generator
from my_data import *
from my_fun.train import train

import moxing as mox
import torch
import numpy as np
import argparse
import torch.distributed as dist
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


def main():
    """
    多机多卡主进程

    :return: None
    """
    parser = argparse.ArgumentParser()

    # 用户设置的模型参数
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--epoch', type=int, default=4, help='epoch')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--k', type=int, default=5, help='passage number of first round retriever')
    parser.add_argument('--m', type=int, default=3, help='passage number of second round retriever')

    # DDP的前四要素
    parser.add_argument('--rank', type=int, default=0, help='index of current task')  # 表示当前是第几个节点
    parser.add_argument('--world_size', type=int, default=2, help='total number of tasks')  # 表示一共有几个节点
    parser.add_argument('--dist_backend', type=str, default='nccl',
                        help='distributed backend')  # model_arts上选好节点数量会自己传过来
    parser.add_argument('--init_method', default=None, help='print process when training')

    args, unparsed = parser.parse_known_args()
    print('args: %s' % args)

    print(f"available GPU numbers:{torch.cuda.device_count()}")
    ngpus_per_node = torch.cuda.device_count() // 1  # 一般来说这里分母是1，即让一个机器上的进程数等于卡数
    args.world_size = ngpus_per_node * args.world_size  # 计算新的进程总数，假设使用12个节点，world_size初始为12,12*8=96,为新的world_size

    # 在主进程中通过torch多进程spawn启动多个子进程
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    # 多进程结束后的操作，用户可以仅在0号节点(rank=0)上进行一些后续操作，比如将子进程运行的中间结果拷贝至云道
    if args.rank == 0:
        print('start to copy model to s3')
        mox.file.copy_parallel(cur_dir + "/end2end_retriever_model_best_f1",
                               s3_dir + '/end2end_retriever_model_best_f1')
        mox.file.copy_parallel(cur_dir + "/end2end_generator_model_best_f1",
                               s3_dir + '/end2end_generator_model_best_f1')
        print('finish copying model to s3')


def main_worker(local_rank, ngpus_per_node, args):
    """
    多机多卡子进程

    :param local_rank: 本地序号
    :param ngpus_per_node: 每个节点的GPU数
    :param args: 参数
    :return: None
    """
    # 先计算global_rank，如果是12个节点，那么global rank的范围从0~95
    global_rank = args.rank * (ngpus_per_node) + local_rank

    # 打印local_rank，这里认为每个节点有几张卡就启动几个进程
    if local_rank is not None:
        print(f"Use GPU: {global_rank} for training")

    # 初始化进程组，需要使用DDP的六要素
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                            rank=global_rank)

    # 认为设置的batch size为192，而情况是12台机器，每台机器8张卡，那每个进程实际分到的数据是2（进程数等于卡数的情况下）
    args.train_batch_size = int(args.train_batch_size / args.world_size)

    # 锁定模型随机种子，保证每个进程的模型初始化相同，因为模型有一些drop out等的随机性的东西，需要seed来达到伪随机
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 加载数据
    train_examples = LoadProcData()
    if local_rank == 0:
        print(f'GPU:{global_rank} | train_examples: {train_examples}\n')

    # 对数据进行distributed sampler,保证每个进程采出的数据不一样
    train_sampler = DistributedSampler(train_examples)
    train_dataloader = DataLoader(dataset=train_examples, sampler=train_sampler, batch_size=args.train_batch_size)

    # 初始化DDP模型，模型分布在不同GPU上
    retriever = Retriever()
    generator = Generator()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    retriever.model.cuda(local_rank)
    generator.model.cuda(local_rank)
    retriever.model = torch.nn.parallel.DistributedDataParallel(retriever.model, device_ids=[local_rank],
                                                                find_unused_parameters=True)
    generator.model = torch.nn.parallel.DistributedDataParallel(generator.model, device_ids=[local_rank],
                                                                find_unused_parameters=True)

    train(epoch=args.epoch, lr=args.lr, k=args.k, m=args.m,
          global_rank=global_rank, local_rank=local_rank,
          retriever=retriever, generator=generator, device=device,
          train_sampler=train_sampler, train_dataloader=train_dataloader)


if __name__ == '__main__':
    print('the end2end_train_roma_DDP script begin running.')
    main()
    print('the end2end_train_roma_DDP script is successfully executed till the end!')
