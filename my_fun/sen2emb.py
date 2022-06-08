import os
import torch
from typing import List
import torch.nn.functional as F
os.system("pip install torchtyping")
from torchtyping import TensorType


def sen2emb(sentences: List[str], owner, device) -> TensorType['len(List[str])', 'hidden vector dimension: 768']:
    """
    将句子批量编码为向量

    :param sentences: 一批句子
    :param owner: 检索器或生成式阅读器
    :param device: CPU或GPU 0或GPU 1或。。。
    :return: 一批句子编码后的张量
    """
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                  min=1e-9)

    encoded_input = owner.tokenizer(sentences, padding=True, truncation=True, max_length=128,
                                    return_tensors='pt').to(device)
    model_output = owner.model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings
