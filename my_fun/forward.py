from my_data import *
from my_fun.sen2emb import sen2emb

import torch
import torch.nn as nn
import numpy as np
from typing import List

os.system("pip install torchtyping")
from torchtyping import TensorType


def forward(q: List[str], a: List[str], k: int, m: int, retriever, generator, device) -> TensorType[1]:
    """
    前向传播，计算loss

    :param q: 一批问题
    :param a: 这批问题对应的一批答案
    :param k: 第一轮，每个问题检索段落数量，受显存限制
    :param m: 第二轮，每个问题和第一轮段落的组合检索段落数量，受显存限制
    :param retriever: 检索器
    :param generator: 生成式阅读器
    :param device: CPU或GPU 0或GPU 1或。。。
    :return: 根据这批q、a数据对的总loss
    """
    bn = len(q)  # 可能数据量除以bs不是整除，意味着可能会有一个的数量不是bs，所以这里定义bn，batch number

    # 第一阶段检索
    q_t1 = q
    q_t1_bar = sen2emb(q_t1, retriever, device)
    I1 = sentences_index.search(q_t1_bar.cpu().detach().numpy(), k)[1]  # index输入维度最多是二维ndarray
    p_t1 = [[sentences[I1[__][_]] for _ in range(k)] for __ in range(bn)]
    p_t1_bar = torch.stack(tuple([sen2emb(p_t1[_], retriever, device) for _ in range(bn)]), 0)
    P1_log = nn.LogSoftmax(dim=-1)(
        torch.sum(q_t1_bar.unsqueeze(1) * p_t1_bar, 2))  # 这一步采用向量数乘，而非双层 for，大大优化时间复杂度; LogSoftmax: 以一种数值稳定的方式进行计算，下同
    # print(f'P1_log:{P1_log}\n')

    # 第二阶段检索
    q_t2_bar = torch.stack(
        tuple([sen2emb([q_t1[__] + ' ' + p_t1[__][_] for _ in range(k)], retriever, device) for __ in range(bn)]), 0)
    I2 = np.array([sentences_index.search(q_t2_bar[_].cpu().detach().numpy(), m)[1] for _ in range(bn)])
    p_t2 = [[[sentences[I2[___][__][_]] for _ in range(m)] for __ in range(k)] for ___ in range(bn)]
    p_t2_bar = torch.stack(
        tuple([torch.stack(tuple([sen2emb(p_t2[__][_], retriever, device) for _ in range(k)]), 0) for __ in range(bn)]),
        0)
    P2_log = nn.LogSoftmax(dim=-1)(torch.sum(q_t2_bar.unsqueeze(2) * p_t2_bar, 3))
    # print(f'P2_log:{P2_log}\n')

    # 联合第一阶段、第二阶段进行检索计算
    P_ret_seqs_log = P1_log.unsqueeze(2) + P2_log  # P_ret_seqs_log: 对于1个问题，给定 k*m 个段落序列时检索器以 log 形式的概率
    # print(f'P_ret_seqs_log:{P_ret_seqs_log}\n')

    # 生成式阅读器计算阶段
    P_gen_seq_log = torch.zeros(bn, k, m).to(device)  # P_gen_seq_log: 对于1个问题，给定 k*m 个段落序列时生成器（生成式阅读器）以 log 形式的概率
    for bi in range(bn):
        for ki in range(k):
            input_seqs = [q[bi] + ' ' + sentences[I1[bi][ki]] + ' ' + sentences[I2[bi][ki][mi]] for mi in range(m)]
            input_ids = generator.tokenizer(input_seqs, return_tensors="pt", padding=True,
                                            truncation=True).input_ids.to(device)
            labels = generator.tokenizer(a[bi], return_tensors="pt").input_ids.repeat(m, 1).to(device)
            output = generator.model(input_ids=input_ids, labels=labels)

            for mi in range(m):
                for token_place in range(len(labels[0])):
                    logits_lsm = nn.LogSoftmax(dim=-1)(output.logits)
                    vocab_place = labels[mi][token_place]  # 一旦token_place由循环给定，那么vocab_place也会被下式确定
                    P_gen_seq_log[bi][ki][mi] += logits_lsm[mi][token_place][vocab_place]
    # print(f'P_gen_seq_log:{P_gen_seq_log}\n')

    # 联合检索器、生成式阅读器计算，这里注意在多机多卡上，loss要利用起模型所有参数，所以加上sum*0
    sums = 0
    for params in retriever.model.parameters():
        sums += torch.sum(params)
    loss = torch.sum(-torch.logsumexp(P_gen_seq_log + P_ret_seqs_log, (1, 2))) + sums * 0
    # print(f'loss:{loss}\n')
    return loss
