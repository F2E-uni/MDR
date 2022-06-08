from my_data import *
from my_fun.sen2emb import sen2emb
from utils import cal_f1_score

import torch
import numpy as np


def predict(k: int, m: int, retriever, generator, device, dataloader) -> float:
    """
    评估模型预测值，这里使用的指标是f1

    :param k: 第一轮，每个问题检索段落数量，受显存限制
    :param m: 第二轮，每个问题和第一轮段落的组合检索段落数量，受显存限制
    :param retriever: 检索器
    :param generator: 生成式阅读器
    :param device: CPU或GPU 0或GPU 1或。。。
    :param dataloader: 数据加载器
    :return: dataloader中所有问题预测答案的平均f1分数
    """
    f1_score_all, answers_pred_all = [], []
    for step, (q, a, qid) in enumerate(dataloader):  # q是问题，a是答案，qid是问题id
        q, a = list(q), list(a)
        bn = len(q)  # 可能数据量除以bs不是整除，意味着可能会有一个的数量不是bs，所以这里定义bn，batch number

        # 第一阶段检索
        q_t1 = q
        q_t1_bar = sen2emb(q_t1, retriever, device)
        I1 = sentences_index.search(q_t1_bar.cpu().detach().numpy(), k)[1]
        p_t1 = [[sentences[I1[__][_]] for _ in range(k)] for __ in range(bn)]

        # 第二阶段检索
        q_t2_bar = torch.stack(
            tuple([sen2emb([q_t1[__] + ' ' + p_t1[__][_] for _ in range(k)], retriever, device) for __ in range(bn)]),
            0)
        I2 = np.array([sentences_index.search(q_t2_bar[_].cpu().detach().numpy(), m)[1] for _ in range(bn)])
        p_t2 = [[[sentences[I2[___][__][_]] for _ in range(m)] for __ in range(k)] for ___ in range(bn)]

        # 检索后生成答案
        generator.tokenizer.padding_side = "left"  # 生成时，我们将使用最右边令牌的 logits 来预测下一个令牌，因此填充应该在左边
        generator.tokenizer.pad_token = generator.tokenizer.eos_token  # 为了避免错误
        input_seqs = [[[q_t1[___] + ' ' + p_t1[___][__] + ' ' + p_t2[___][__][_]
                        for _ in range(m)] for __ in range(k)] for ___ in range(bn)]

        f1_score, answers_pred_final = [0] * bn, [' '] * bn
        for bi in range(bn):
            for ki in range(k):
                inputs = generator.tokenizer(input_seqs[bi][ki], return_tensors="pt", padding=True)
                inputs["input_ids"] = inputs["input_ids"].to(device)
                inputs["attention_mask"] = inputs["attention_mask"].to(device)
                output_seqs = generator.model.module.generate(  # 因为分布式中model外面包了一个DistributedDataParallel，所以要加.module
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=False,  # 禁用采样以测试 batching 是否影响输出
                )
                answers_pred = generator.tokenizer.batch_decode(output_seqs, skip_special_tokens=True)

                # 根据生成的答案和答案label计算f1分数
                for mi in range(m):
                    f1_score_temp = cal_f1_score(answers_pred[mi], a[bi])[0]
                    if f1_score_temp >= f1_score[bi]:
                        f1_score[bi] = f1_score_temp
                        answers_pred_final[bi] = answers_pred[mi]

        # print(
        #     f'question index: {list(qid)}\n'
        #     f'question: {list(q)}\n'
        #     f'f1 score: {f1_score}\n'
        #     f'answers prediction: {answers_pred_final}\n')
        f1_score_all += f1_score
        answers_pred_all += answers_pred_final

    return sum(f1_score_all) / len(f1_score_all)
