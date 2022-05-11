import codecs
import json
import random

data_dirs = ['data/annoted/A股交易规则/span_fqa.json',
             'data/annoted/A股新股/span_fqa.json',
             'data/annoted/创业板/span_fqa.json',
             'data/annoted/北交所-test/span_fqa.json',
             'data/annoted/港股通/span_fqa.json',
             'data/annoted/行政事务办事指南汇编V5.0/span_fqa.json',
             'data/annoted/预约打新/span_fqa.json',
             'data/annoted/other_code/span_fqa.json']

train_dir = 'data/train.json'
dev_dir = 'data/dev.json'
data_list = []
global_id = 0
split_rate = 0.9

for data_dir in data_dirs:
    with codecs.open(data_dir, "r", "utf-8") as fr:
        data_dicts = json.load(fr)
        for data_dict in data_dicts:
            question = ''
            answer = ''
            answer_start = -1
            context = ''
            for key, value in data_dict.items():
                if key != 'info':
                    question = key
                    if len(value) == 0:
                        break
                    answer = value[0]
                else:
                    context_infos = value
                    for context_info in context_infos:
                        context = context_info['context']
                        answer_start = context_info['span'][0]

                        data_list.append({'id': '{}'.format(global_id), 'question': question, 'context': context, 'answers': {'answer_start': [int(answer_start)], 'text': [answer]}})
                        global_id += 1

random.shuffle(data_list)
train_size = int(len(data_list) * split_rate)
train_data = data_list[:train_size]
dev_data = data_list[train_size:]
with codecs.open(train_dir, "w", "utf-8") as fw1:
    json.dump({'version': 1.0, 'data': train_data}, fw1, ensure_ascii=False)
with codecs.open(dev_dir, "w", "utf-8") as fw2:
    json.dump({'version': 1.0, 'data': dev_data}, fw2, ensure_ascii=False)
