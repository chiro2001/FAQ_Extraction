
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import docx
import re
import os
import json
import copy
import numpy as np


def clean(str):
    #清理文本中误匹配的标记信息
    pattern1 = re.compile(r"<A\d+>")
    pattern2 = re.compile(r"<\\A\d+>",re.S)
    bad = pattern1.findall(str)
    bad+=pattern2.findall(str)
    # print(bad)
    for b in bad:
        str = str.replace(b,'')
    return str

 
def iter_block_items(parent):

    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")
 
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield (Paragraph(child, parent), 1)
        elif isinstance(child, CT_Tbl):
            # Table(child, parent)
            table = Table(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        yield (paragraph,0)


def doc2text(file_path):      
    

    doc = docx.Document(file_path)
        
    texts = ''
    for block,typ in iter_block_items(doc):
        # if typ ==1:
        para = ''
        for run in block.runs:
            para +=run.text
        texts +=para+'\n'

    return texts



def rules(context):
    #! 定义一系列规则
    head1 = ['一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四','十五','十六','十七']
    for i,x in enumerate(head1):
        head1[i] = x+'、'

    head2=['（'+h.replace('、','')+'）' for h in head1]

    head3 = [str(i)+'、' for i in range(1,20)]

    head4= ['（'+str(i)+'）' for i in range(1,20)]

    head5 = []
    head6 = []
    if '1）' in context:
        head5 = [str(i)+'）' for i in range(1,20)]
    else:
        head5 = []
    if '①' in context:
        head6 = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩']
    else:
        head6 = []
    heads = [head1, head2, head3, head4, head5, head6]
    return heads


def get_segments(lines):

    def clean_pre(pre):
        t = pre
        t = t.replace('\n','')
        for head in heads:
            for h in head:
                # print(h)
                t = t.replace(h, '')
        return t


    
    heads = rules(''.join(lines))
    
    seps = []
    
    for head in heads:
        #从上级标题处理到下一级
        inds = []
        for i, l in enumerate(lines):
            for h in head:
                if l.startswith(h):
                    inds.append(i)
        if len(inds)!=0:
            seps.append(inds)

    results = []
    def get_result(beg, end, nlevel,level, pre):


        if level == len(seps)-1:
            # for i in range(len(seps[level])-1):
            #     results.append(''.join(lines[seps[level][i]:seps[level][i+1]]))
            return
        
        results.append(''.join(lines[beg:end]))
        
        arg = copy.deepcopy(level)
        arg += 1
        
        for i in range(len(nlevel) - 1):
            nex = [x for x in seps[arg] if x > nlevel[i] and x < nlevel[i+1]]
            ppre = pre+clean_pre(lines[beg])+','

            get_result(nlevel[i], nlevel[i+1], nex, arg,ppre)
        
        l = len(nlevel)
        if l!=0:
            nex = [x for x in seps[arg] if x > nlevel[l-1] and x < end]
            ppre = pre + clean_pre(lines[beg])+','
            
            get_result(nlevel[l-1], end, nex, arg, ppre)
            
    get_result(0, len(lines), seps[0], 0, '关键词:')
    
    segments = "\n\n".join(results[:-1])
        
    return segments

import pandas as pd

if __name__=='__main__':

    config_path = './file_config'
    source_file_paths = []
    target_file_paths = []
    with open(config_path,'r',encoding='utf-8') as f:
        s = f.readline().strip()
        t = f.readline().strip()
        x = f.readline()
        source_file_paths.append(s)
        target_file_paths.append(t)

    for source_path, target_path in zip(source_file_paths, target_file_paths):
        
        txt = doc2text(source_path)
        

        segments  = get_segments(txt.replace('\xa0','').split('\n'))
        segments = list(set(segments.split('\n\n')))
        segs = pd.DataFrame()
        segs['answer'] = pd.Series(segments)
        segs.to_csv(target_path, index=None,encoding='utf-8')
        
        
        

