
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


def doc2text(file_path,klass = 'context'):      
    
    out = "/".join(file_path.split('/')[:-1]) + f"/{klass}_pure_text.txt"
      
    doc = docx.Document(file_path)
        
    texts = ''
    for block,typ in iter_block_items(doc):
        # if typ ==1:
        para = ''
        for run in block.runs:
            para +=run.text
        texts +=para+'\n'

                
    # texts = texts.replace()
    
    with open(out,'w',encoding='utf-8') as fw:
        fw.write(texts)
    if klass=='context':
        with open(out+'_notag','w',encoding='utf-8') as fw:
            fw.write(clean(texts))

    return clean(texts), texts

def get_answers(context, questions, path):                
    answers = []
    for i in range(len(questions)):
        
        pattern = re.compile(r"<A{}>(.+?)<\\A{}>".format(str(i+1),str(i+1)),re.S)

        # print(i+1)
        result = pattern.findall(context)
        for i_a,a in enumerate(result) :
            result[i_a] =clean(a)

        answers.append(list(set(result)))
        
        if len(result)==0:
            print(i+1)
            
    assert len(answers) == len(questions)
    fqa = [{k.strip():v} for k,v in zip(questions, answers)]
    with open(os.path.join(path, 'FQA.json'),'w',encoding='utf-8') as f:
        json.dump(fqa,f,ensure_ascii=False)
    return fqa


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


def get_segments(file, data_path):
    
    out_path = os.path.join(data_path,'segments.txt')
    
    def clean_pre(pre):
        t = pre
        t = t.replace('\n','')
        for head in heads:
            for h in head:
                # print(h)
                t = t.replace(h, '')
        return t

    with open(file,'r',encoding='utf-8') as f:
        lines = f.readlines()
    
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
        
        results.append(pre+'\n'+''.join(lines[beg:end]))
        
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
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(segments)
        
    return segments

def get_spans(fqa,data_path, all_paras):
    out = os.path.join(data_path, 'span_fqa.json')
    for i_qa,qa in enumerate(fqa):
        ans = ''.join([v[0] for v in qa.values() if len(v) !=0])
        ans = ans.replace('\n\n','\n')
        if len(ans)==0:
            continue
        match = []
        for para in all_paras:
            start = para.find(ans)
            end = -1
            if start !=-1:
                end = start + len(ans)-1
                s = np.random.choice( max(int(start / 4), 1),1)[0]
                e = np.random.choice(max(int(( len(para) - end) / 4), 1),1)[0]
                    
                print(start-s , end+e)
                    
                match.append({'context':para[start-s : end+e+1],'span':[str(s),str(s + len(ans)-1)]})
                # else:
                #     match.append({'context':para,'span':[str(start),str(end)]})
                    
                        
        qa['info'] = match
        
        fqa[i_qa] = qa
    with open(out,'w',encoding='utf-8') as f:
        json.dump(fqa,f,ensure_ascii=False)

if __name__=='__main__':

    doc_file_path = [
        '/home/wangrui/2022项目/annoted/北交所-test/北交所新股申购业务汇总.docx', 
                     '/home/wangrui/2022项目/annoted/港股通/港股通开通业务汇总.docx',
                     '/home/wangrui/2022项目/annoted/创业板/创业板业务汇总.docx',
                     '/home/wangrui/2022项目/annoted/行政事务办事指南汇编V5.0/行政事务办事指南汇编V5.0.docx',
                     '/home/wangrui/2022项目/annoted/预约打新/预约打新业务汇总.docx',
                     '/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则.docx',
                     '/home/wangrui/2022项目/annoted/A股新股/普通账户新股申购（A股）业务汇总.docx'
                     ]
    #文件原文
    ques_file_path =[
        '/home/wangrui/2022项目/annoted/北交所-test/北交所新股申购业务汇总Question.docx',
        '/home/wangrui/2022项目/annoted/港股通/港股通开通业务汇总 Question.docx',
        '/home/wangrui/2022项目/annoted/创业板/创业板业务汇总 Question.docx',
        '/home/wangrui/2022项目/annoted/行政事务办事指南汇编V5.0/行政事务办事指南汇编V5.0.docx',
        '/home/wangrui/2022项目/annoted/预约打新/预约打新业务汇总 Question.docx',
        '/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则 Question.docx',
        '/home/wangrui/2022项目/annoted/A股新股/新建 Microsoft Word 文档.docx'
                     ]
    #标注问题原文
    #文件目录需要包含docx文件
    for txt_path, ques_path in zip(doc_file_path, ques_file_path):
        
        txt, tag_txt = doc2text(txt_path,'context')
        
        ques, _ = doc2text(ques_path,'question')
        data_path = '/'.join(txt_path.split('/')[:-1])
        #! 以上将docx的文本和问题转换为纯文本格式
        #! 下面开始答案抽取
        
        fqa = get_answers(tag_txt, ques.split('\n'), data_path)
        
        #! 下面将文本按照标题拆分
        segments  = get_segments(os.path.join(data_path, 'context_pure_text.txt_notag'), data_path)
        segments = list(set(segments.split('\n\n')))
        
        #! 文本按照段落拆分
        with open(os.path.join(data_path, 'context_pure_text.txt_notag'),'r',encoding='utf-8') as f:
            lines = f.readlines()
        lines = list(set(lines))
        
        all_paras= segments +lines        
        
        get_spans(fqa, data_path, all_paras)
                
        
        

