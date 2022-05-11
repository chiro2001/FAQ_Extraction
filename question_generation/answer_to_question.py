import keras.backend.tensorflow_backend as KTF
import numpy as np
import pandas as pd
import tensorflow as tf
import ipykernel
import os
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.layers import Loss
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import Model
from rouge import Rouge  # pip install rouge
from sklearn.model_selection import KFold
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils import json2df, preprocess, preprocess_no_passage


# os.environ["TF_KERAS"]='1'

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
    df = pd.read_csv(filename,encoding="UTF-8")# csv转DataFrame
    df = preprocess_no_passage(df)     # 数据预处理

    # 文本截断
    D = list()
    for _, row in df.iterrows():
        question = row['question']
        answer = row['answer']
        if len(answer) < max_a_len -2:
            D.append((answer, question))
        else:
            a = answer[:max_a_len-2]
            D.append((a, question))
    return D


class data_generator(DataGenerator):
    """数据生成器。"""
    def __init__(self, data, batch_size=32, buffer_size=None, random=False):
        super().__init__(data, batch_size, buffer_size)
        self.random = random

    def __iter__(self, random=False):
        """单条样本格式：[CLS]答案[SEP]问题[SEP]
        """
        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for is_end, (a, q) in self.sample(random):

            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = a_token_ids + q_token_ids[1:]
            segment_ids = [0] * len(a_token_ids)
            segment_ids += [1] * (len(token_ids) - len(a_token_ids))
            # 缓解暴露偏置问题
            o_token_ids = token_ids
            if np.random.random() > 0.5:
                token_ids = [
                    t if s == 0 or (s == 1 and np.random.random() > 0.3)
                    else np.random.choice(token_ids)
                    for t, s in zip(token_ids, segment_ids)
                ]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_o_token_ids.append(o_token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_o_token_ids = sequence_padding(batch_o_token_ids)
                yield [batch_token_ids, batch_segment_ids, batch_o_token_ids], None
                batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d


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
    # 交叉熵作为loss，并mask掉输入部分的预测
    y_true = train_model.input[2][:, 1:]  # 目标tokens
    y_mask = train_model.input[1][:, 1:]
    y_pred = train_model.output[0][:, :-1]  # 预测tokens，预测与目标错开一位
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

    train_model.add_loss(cross_entropy)
    train_model.compile(optimizer=Adam(1e-5))

    return model,train_model


def adversarial_training(model, embedding_name, epsilon=1.):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数



class QuestionGeneration(AutoRegressiveDecoder):
    """通过beam search来生成问题。"""
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @AutoRegressiveDecoder.wraps(default_rtype='pronas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.model.predict([token_ids, segment_ids])[:, -1]


    def generate(self, answer, topk):
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = a_token_ids
        segment_ids = [0] * len(a_token_ids)
        q_ids = self.beam_search([token_ids, segment_ids], topk=topk)
        return tokenizer.decode(q_ids)


class Evaluator(keras.callbacks.Callback):
    """计算验证集rouge_l。"""
    def __init__(self, valid_data, qg, topk):
        super().__init__()
        self.rouge = Rouge()
        self.best_rouge_l = 0.0
        self.valid_data = valid_data
        self.qg = qg
        self.topk = topk
        self.smooth = SmoothingFunction().method1

    def on_epoch_end(self, epoch, logs=None):
        result = self.evaluate(self.valid_data, self.topk)
        rouge_l = result["rouge-l"]  # 评测模型
        if rouge_l > self.best_rouge_l:
            self.best_rouge_l = rouge_l
        logs['val_rouge_l'] = rouge_l
        print(
            f'val_rouge_l: {rouge_l:.5f}, '
            f'best_val_rouge_l: {self.best_rouge_l:.5f}',
            f'val_rouge_1:{result["rouge-1"]:.5f}',
            f'val_rouge_2:{result["rouge-2"]:.5f}',
            f'bleu:{result["bleu"]:.5f}',
            end=''
        )
    # topk=1 beam search
    def evaluate(self, data, topk):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for a, q in tqdm(data):
            total += 1
            q = ' '.join(q)
            pred_q = ' '.join(self.qg.generate(answer=a, topk=topk))
            if pred_q.strip():
                scores = self.rouge.get_scores(hyps=pred_q, refs=q)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[q.split(' ')],
                    hypothesis=pred_q.split(' '),
                    smoothing_function=self.smooth
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total

        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }


def predict_to_file(data, filename, qag, topk):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            q = qag.generate(d[0], topk=topk)
            s = '%s\t%s\n' % (q, d[0])
            f.write(s)
            f.flush()


def do_train():
    data = load_data('./data/SogouQA_SingleQuestion.csv')  # 加载数据

    # 交叉验证
    kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data), 1):
        print(f'Fold {fold}')

        # 配置Tensorflow Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        sess = tf.Session(config=config)
        KTF.set_session(sess)

        # 划分训练集和验证集
        train_data = [data[i] for i in trn_idx]
        valid_data = [data[i] for i in val_idx]

        train_generator = data_generator(train_data, batch_size, random=True)
        model, train_model = build_model()  # 构建模型

        adversarial_training(train_model, 'Embedding-Token', 0.5)  # 对抗训练

        # 问题生成器
        qg = QuestionGeneration(
            model, start_id=None, end_id=tokenizer._token_dict['？'],
            maxlen=max_q_len
        )

        # 设置回调函数
        callbacks = [
            Evaluator(valid_data, qg, topk=topk),
            EarlyStopping(
                monitor='val_rouge_l',
                patience=3,
                verbose=1,
                mode='max'),
            ModelCheckpoint(
                f'./saved_model/baseline-sougou-fold-{fold}.h5',
                monitor='val_rouge_l',
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
                mode='max'),
            TensorBoard(log_dir='./output/sougou_logs', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False,
                        write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                        embeddings_data=None, update_freq='epoch')
        ]

        # 模型训练
        print("Training")
        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=callbacks,
        )

        # predict_to_file(valid_data, './prediction_result/qa_sougou_predict.csv', qg, topk)

        KTF.clear_session()
        sess.close()
        break #只取第一折的模型


if __name__ == '__main__':
    do_train()




















