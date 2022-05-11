
from docx import Document
#读取文档

files = ['/home/wangrui/2022项目/annoted/A股交易规则/A股交易规则.docx',
         "/home/wangrui/2022项目/annoted/创业板/创业板业务汇总.docx",
         "/home/wangrui/2022项目/annoted/港股通/港股通开通业务汇总.docx"]


from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph
import docx
import openpyxl
import xlsxwriter
 
def iter_block_items(parent):
    """
    Yield each paragraph and table child within *parent*, in document order.
    Each returned value is an instance of either Table or Paragraph. *parent*
    would most commonly be a reference to a main Document object, but
    also works for a _Cell object, which itself can contain paragraphs and tables.
    """
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

for f in files:
    
    doc = docx.Document(f)
    
    texts = ''
    for block,typ in iter_block_items(doc):
        # if typ ==1:
        para = ''
        for run in block.runs:
            para +=run.text
        texts +=para+'\n'

            
    # texts = texts.replace()
    out = "/".join(f.split('/')[:-1]) + f"/pure_text.txt"
    with open(out,'w',encoding='utf-8') as fw:
        fw.write(texts)
    with open(out+'_notag','w',encoding='utf-8') as fw:
        fw.write(clean(texts))
    
    #     # pass
    # if block.style.name == 'Heading 1':
    #     # pass
    #     print(block)
 
 