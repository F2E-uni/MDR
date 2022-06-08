from my_global_vars import *

import os
import re
import string
import moxing as mox
from collections import Counter

os.system("pip install nltk")
# 安装model_arts上缺少的包
os.system(
    "cd /home/ma-user; mkdir nltk_data; "
    "cd nltk_data; mkdir tokenizers; mkdir corpora; "
    "cd tokenizers; mkdir punkt; "
    "cd ../corpora; mkdir wordnet; mkdir omw-1.4")  # 注意这里不能分开写，因为一个os.system相当于一个进程；且注意mkdir不能跨级创建
# 官方推荐的nltk安装方式不是pip，用pip安装需要补充离线的子包
mox.file.copy_parallel(cur_dir + '/utils/nltk/punkt', '/home/ma-user/nltk_data/tokenizers/punkt')  # 注意是A文件夹下的全部copy到B文件夹下
mox.file.copy_parallel(cur_dir + '/utils/nltk/wordnet', '/home/ma-user/nltk_data/corpora/wordnet')
mox.file.copy_parallel(cur_dir + '/utils/nltk/omw-1.4', '/home/ma-user/nltk_data/corpora/omw-1.4')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def cal_f1_score(prediction: str, ground_truth: str) -> (float, float, float):
    """
    计算两个句子的f1分数

    :param prediction: 模型根据问题预测出的答案
    :param ground_truth: 该问题对应的正确答案
    :return f1, precision, recall: 分别对应预测值和真实值得f1分数，准确率和召回率
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def normalize_answer(s: str) -> str:
    """
    对输入句子进行处理，返回正则化后的句子

    :param s: 句子
    :return: 正则化后的句子
    """

    def remove_articles(text: str) -> str:
        """
        :param text: 句子
        :return: 去除冠词a，an，the后的句子
        """
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        """
        :param text: 句子
        :return: 去除多余空格后的句子（因为单词和单词之间可能会出现用多个空格分割的情况）
        """
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        """
        :param text: 句子
        :return: 去除标点符号后的句子
        """
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """
        :param text: 句子
        :return: 全部转小写后的句子
        """
        return text.lower()

    def stemming(sentence: str) -> str:
        """
        :param sentence: 句子
        :return: 对每个单词进行词干提取后的句子。词干提取：如cats变为cat
        """
        ps = PorterStemmer()
        words = word_tokenize(sentence)
        sentence_done = ''
        for w in words:
            sentence_done = sentence_done + ' ' + ps.stem(w)
        return sentence_done

    def lemmatization(sentence: str) -> str:
        """
        :param sentence: 句子
        :return: 对每个单词进行词性还原后的句子。词干提取：如drove变为drive（比词干提取更复杂些）
        """
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(sentence)
        sentence_done = ''
        for w in words:
            sentence_done = sentence_done + ' ' + lemmatizer.lemmatize(w)
        return sentence_done

    return lemmatization(stemming(white_space_fix(remove_articles(remove_punc(lower(s))))))
