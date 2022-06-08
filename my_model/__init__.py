from my_global_vars import *

os.system("pip install transformers==4.18.0")
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration


class Retriever:
    def __init__(self):
        # copy_parallel命令填的两个参数都是文件夹，意思是把前面参数文件夹中的所有东西copy到后面参数的文件夹中
        self.tokenizer = AutoTokenizer.from_pretrained(cur_dir + '/model/all-mpnet-base-v2')  # GPU训练，tokenizer不用放到gpu上
        self.model = AutoModel.from_pretrained(cur_dir + '/model/all-mpnet-base-v2')  # GPU训练，model要放到GPU上


class Generator:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(cur_dir + "/model/t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(cur_dir + "/model/t5-base")
