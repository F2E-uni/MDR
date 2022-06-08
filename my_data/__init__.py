from my_global_vars import *

from torch.utils.data import Dataset
import os

os.system("pip install faiss-gpu==1.7.2")
import faiss

os.system("pip install pickle5")
import pickle5 as pickle


class LoadProcData(Dataset):
    def __init__(self):
        self.questions = []
        self.answers = []
        self.questions_index = []
        self.load_data()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx], self.questions_index[idx]

    # 注：此时只load数据，不tokenize数据，因为第一：创建LoadProcData时还未定义tokenizer；第二：数据可能经过的是检索器或阅读器的tokenizer，并非单一
    def load_data(self):
        with open(cur_dir + '/data/animals_full_formatted.csv', 'r') as f:
            for line in f.readlines()[1:]:  # 这里从第二行开始是因为首行是表头，不是数据
                items = line.split('\t')
                self.questions.append(items[0])  # 这里的索引值0根据csv文件的格式来
                self.answers.append(items[4].replace('\n', ''))  # 这里的索引值4根据csv文件的格式来
        self.questions_index = list(range(len(self.questions)))
        print(f'complete loading questions and answers! the length of questions or answers is {len(self.questions)}')


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


sentences_index = faiss.read_index(cur_dir + '/data/sentences.index')
sentences = pickle.load(open(cur_dir + '/data/animals_sentences.pickle', 'rb'))
print(f'complete loading sentences! The length of sentences is {len(sentences)}\n')
