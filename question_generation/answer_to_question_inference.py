import json
import os
import re
import pandas as pd
import numpy as np
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import AutoRegressiveDecoder
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from bert4keras.layers import Loss
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import preprocess_no_passage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 指定GPU

# 基本参数
n = 5               # 交叉验证
max_q_len = 66     # 问题最大长度
max_a_len = 170      # 答案最大长度
batch_size = 4      # 批大小
epochs = 20         # 迭代次数
SEED = 2022         # 随机种子
topk = 1            # beam search topk

# nezha配置
config_path = './pre_trained_model/NEZHA-Base-WWM/bert_config.json'
checkpoint_path = './pre_trained_model/NEZHA-Base-WWM/model.ckpt-691689'
dict_path = './pre_trained_model/NEZHA-Base-WWM/vocab.txt'

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def load_data(filename):
    """加载数据。"""
    df = pd.read_csv(filename,encoding="UTF-8", on_bad_lines='skip')# csv转DataFrame
    df = preprocess_no_passage(df)     # 数据预处理

    # 文本截断
    D = list()
    for _, row in df.iterrows():
        # passage = row['passage']
        question = row['question']
        answer = row['answer']
        document = row['document']
        if len(answer) < max_a_len -2:
            D.append((answer, question, document))
        else:
            a = answer[:max_a_len-2]
            D.append((a, question, document))

    return D


def build_model():
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model="nezha",
        application='unilm',
        keep_tokens=keep_tokens, # 只保留keep_tokens中的字，精简原字表
    )
    o_in = Input(shape=(None, ))
    train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

    return model,train_model


class QuestionGeneration(AutoRegressiveDecoder):
    """通过beam search来生成问题。"""
    def __init__(self, models, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = models

    @AutoRegressiveDecoder.wraps(default_rtype='pronas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        # 五折中选择概率最高的作为预测结果
        probas = list()
        for i in range(n):
            proba = self.models[i].predict([token_ids, segment_ids])[:, -1]
            probas.append(proba)

        return np.mean(np.concatenate(probas), axis=0, keepdims=True)

    def generate(self, answer, topk):
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = a_token_ids
        segment_ids = [0] * len(a_token_ids)
        q_ids = self.beam_search([token_ids, segment_ids], topk=topk)
        return tokenizer.decode(q_ids)


def predict_to_file(data, filename, qag, topk):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        s = '%s,%s,%s,%s\n' %("predict_question", "truth_question", "answer", "document")
        f.write(s)
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q = qag.generate(d[0], topk=topk)
            s = '%s,%s,%s,%s\n' % (q, d[1], d[0], d[2])
            f.write(s)
            f.flush()


def do_infer(input_path, output_path):

    # 加载模型权重
    models = []
    for i in range(1, n + 1):
        model, train_model = build_model()
        train_model.load_weights(f'./saved_model/baseline-sougou-zhaoshang-fold-{i}.h5')
        models.append(model)

    qag = QuestionGeneration(
        models, start_id=None, end_id=tokenizer._token_dict['？'],
        maxlen=max_q_len
    )
    test_data = load_data(input_path)
    predict_to_file(test_data, output_path, qag, topk=topk)


if __name__ == '__main__':
    input_path = "./data/test.csv"
    output_path = "./prediction_result/test_predict_sougou-zhaoshang-fold_1-5_least_data_least_model.csv"
    do_infer(input_path=input_path, output_path=output_path)