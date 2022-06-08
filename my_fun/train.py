from my_global_vars import *
from my_fun.forward import forward  # 在这里不能from forward import forward，model_arts上只看相对pwd
from my_fun.predict import predict

from torch import optim


def train(epoch, lr, k, m, retriever, generator, device, global_rank, local_rank, train_sampler,
          train_dataloader) -> None:
    """
    训练模型

    :param epoch: 迭代次数
    :param lr: 学习率learning rate
    :param k: 第一轮，每个问题检索段落数量，受显存限制
    :param m: 第二轮，每个问题和第一轮段落的组合检索段落数量，受显存限制
    :param retriever: 检索器
    :param generator: 生成式阅读器
    :param device: CPU或GPU 0或GPU 1或。。。
    :param global_rank: 多机多卡中的全局编号，如3*8+2=26中的26（3是机器数，8是每台机器卡数）
    :param local_rank: 多机多卡中的当前编号，如3*8+2=26中的2（3是机器数，8是每台机器卡数）
    :param train_sampler: 数据随机切片取样方式
    :param train_dataloader: 数据加载器
    :return: None，模型参数得到训练
    """
    # 定义优化器
    optimizer = optim.Adam([
        {'params': retriever.model.parameters(), 'lr': lr},
        {'params': generator.model.parameters(), 'lr': lr}
    ])

    f1_avg_highest = 0
    for e in range(epoch):  # e代表对epoch的迭代
        train_sampler.set_epoch(e)  # 保证每个epoch启动random
        for step, (q, a, qid) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = forward(q=list(q), a=list(a), k=k, m=m, retriever=retriever, generator=generator, device=device)
            loss.backward()
            optimizer.step()
            if local_rank == 0:  # 仅在global_rank=0时打印loss，否则每个进程会打印一份，由于数据输入不同，每个进程的loss不同
                print(f'GPU:{global_rank} | epoch:{e + 1} | step:{step + 1} | loss_avg:{loss / len(q):.4f} | '
                      f'qid:{tuple(qid.tolist())}\n')
                # print(f'q:{q} | a:{a}\n')

            # # 可以打印每个进程的梯度，查看梯度和模型参数是否完全相同
            # if local_rank == 0 or local_rank == 1 or local_rank == 2:
            #     print(f'next(retriever.model.parameters()):{next(retriever.model.parameters())} | '
            #                  f'next(generator.model.parameters()):{next(generator.model.parameters())}\n')

        # 计算预测f1
        f1_avg_temp = predict(k=k, m=m, retriever=retriever, generator=generator, device=device,
                              dataloader=train_dataloader)
        if local_rank == 0:
            print(f"GPU:{global_rank} | epoch:{e + 1} | f1_avg:{f1_avg_temp:.4f}\n")

        # 根据当前epoch后模型的f1值是否比上一个epoch高，来选择是否保存当前模型到best目录
        if f1_avg_temp > f1_avg_highest:
            f1_avg_highest = f1_avg_temp
            if global_rank == 0:
                # 注意：这里只是保存到model arts上，由于s3和model arts文件映射的单向性，在DDP脚本的主进程中还需执行mox.copyparallel
                retriever.model.module.save_pretrained(cur_dir + "/end2end_retriever_model_best_f1")
                generator.model.module.save_pretrained(cur_dir + "/end2end_generator_model_best_f1")
                print(f"GPU:{global_rank} | epoch:{e + 1} | saved the best model to cur directory!\n")

        # 为了log中输出更好看些，将该次epoch和上次epoch隔开
        if local_rank == 0:
            print(f"GPU:{global_rank} | -------------------------------------\n")
