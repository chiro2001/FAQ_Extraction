import json

json_file ='/home/wangrui/2022项目/annoted/A股交易规则/FQA.json'
with open(json_file,'r',encoding='utf-8') as f:

    fqa = json.load(f)

# print(fqa)

seg_file = '/home/wangrui/2022项目/annoted/A股交易规则/segments.txt'
def getSegs(file):
    result = []
    with open(file,'r',encoding='utf-8') as f:
        str = ''
        for st in f.readlines():
            if st!='\n':
                str+=st
            else:
                result.append(str)
                str = ''
    return result

text_file = "/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则_pure_text.txt_notag"
def getParas(file):
    result = []
    with open(file,'r',encoding='utf-8') as f:
        for st in f.readlines():
            result.append(st)
    return result

# segs = getSegs(seg_file)

# texts =getParas(text_file)


segs = list(set(getSegs(seg_file)))

texts = list(set(getParas(text_file)))

all_paras = segs

all_paras= segs +texts
import numpy as np

print(len(fqa))

for i_qa,qa in enumerate(fqa):
    for v in qa.values():
        print(v,type(v))
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
            # tmp = {'context':para[start:end],'span':[start,end]}
            if start > 256:
                s = np.random.choice(max(int(start / 2), 1),1)[0]
                e = np.random.choice(max(int(( len(para) - end) / 2), 1),1)[0]
                print(s, e)
                match.append({'context':para[start-s : end+e],'span':[str(s),str(s + len(ans)-1)]})

            else:
                match.append({'context':para[start:end],'span':[str(start),str(end)]})
                
                    
    qa['info'] = match
    # print(match)
    print(i_qa)
    fqa[i_qa] = qa
print(fqa)
with open("/".join(text_file.split('/')[:-1])+'/span_fqa.json','w',encoding='utf-8') as f:
    # f.write(str(fqa))
    json.dump(fqa,f,ensure_ascii=False)

