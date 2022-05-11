

head1 = ['一','二','三','四','五','六','七','八','九','十','十一','十二','十三','十四','十五','十六','十七']
for i,x in enumerate(head1):
    head1[i] = x+'、'

head2=['（'+h.replace('、','')+'）' for h in head1]

head3 = [str(i)+'、' for i in range(1,20)]

head4= ['（'+str(i)+'）' for i in range(1,20)]

head5 = []
head6 = []
# head5 = [str(i)+'）' for i in range(1,20)]

# head6 = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩']

file = "/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则_pure_text.txt_notag"

out = '/'.join(file.split('/')[:-1])+'/'+'segments.txt'

heads = [head1, head2, head3, head4, head5, head6]

with open(file,'r',encoding='utf-8') as f:
    lines = f.readlines()

seps = []
flag = [True for i in range(len(lines))]
for head in heads:
    #从上级标题处理到下一级
    inds = []
    for i, l in enumerate(lines):
        for h in head:
            if l.startswith(h):
                inds.append(i)
    if len(inds)!=0:
        seps.append(inds)

seps.append([5])
print(seps)

results = []
import copy
def clean(pre):
    t = pre
    t = t.replace('\n','')
    for head in heads:
        for h in head:
            # print(h)
            t = t.replace(h, '')
    return t


def get_result(beg, end, nlevel,level, pre):


    if level == len(seps)-1:
        # for i in range(len(seps[level])-1):
        #     results.append(''.join(lines[seps[level][i]:seps[level][i+1]]))
        return
    
    results.append(pre+'\n'+''.join(lines[beg:end]))
    
    arg = copy.deepcopy(level)
    arg += 1
    # if arg ==6:
    #     return
    for i in range(len(nlevel) - 1):
        nex = [x for x in seps[arg] if x > nlevel[i] and x < nlevel[i+1]]
        ppre = pre+clean(lines[beg])+','

        get_result(nlevel[i], nlevel[i+1], nex, arg,ppre)
    
    l = len(nlevel)
    if l!=0:
        nex = [x for x in seps[arg] if x > nlevel[l-1] and x < end]
        ppre = pre + clean(lines[beg])+','
        
        get_result(nlevel[l-1], end, nex, arg, ppre)
        

get_result(0, len(lines), seps[0], 0, '关键词:')


with open(out, 'w', encoding='utf-8') as f:
    for r in results[1:]:
        if True:
            f.write(r)
            f.write('\n\n')



