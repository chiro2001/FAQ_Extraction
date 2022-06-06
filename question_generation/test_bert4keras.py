from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
import numpy as np

# config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'
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

# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
# model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model="nezha",
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')

print('\n ===== predicting =====\n')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))
