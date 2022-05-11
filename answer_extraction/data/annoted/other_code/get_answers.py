


question_f = '/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则 Question_pure_text.txt'
with open(question_f,'r',encoding='utf-8') as f:
    questions = f.readlines()


text_f = '/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则_pure_text.txt'
with open(text_f, 'r', encoding='utf-8') as f:
    text = '\n'.join(f.readlines())

import pandas as pd
import re


def clean(str):
    pattern1 = re.compile(r"<A\d+>")
    pattern2 = re.compile(r"<\\A\d+>",re.S)
    bad = pattern1.findall(str)
    bad+=pattern2.findall(str)
    print(bad)
    for b in bad:
        str = str.replace(b,'')
    return str
import json

answers = []
for i in range(len(questions)):
    
    pattern = re.compile(r"<A{}>(.+?)<\\A{}>".format(str(i+1),str(i+1)),re.S)

    # print(i+1)
    result = pattern.findall(text)
    for i_a,a in enumerate(result) :
        result[i_a] =clean(a)

    answers.append(list(set(result)))
    if len(result)==0:
        print(i+1)
assert len(answers) == len(questions)
fqa = [{k.strip():v} for k,v in zip(questions, answers)]
with open('/'.join(question_f.split('/')[:-1])+'/FQA.json','w',encoding='utf-8') as f:
    json.dump(fqa,f,ensure_ascii=False)
