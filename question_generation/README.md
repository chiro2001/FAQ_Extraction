# 招商问题生成模块

## 简介

### 环境

- `bert4keras`==0.10.9
- `keras`==2.3.1
- `tensorflow`==1.15.4
- `tensorboard`==1.15.0

### 文件说明

```yaml
D:.
│  answer_to_question.py
│  answer_to_question_fine_tuning.py
│  answer_to_question_inference.py
│  metric.py
│  README.md
│  utils.py
│
├─data
│      SogouQA_SingleQuestion.csv
│      test.csv
│      train.csv
│
├─raw_doc
│	北交所(源文件).docx
|
├─middle
│	北交所(文件处理中间结果).csv
|
├─output
│  ├─sougou_logs
│  └─zhaoshang_logs
│          events.out.tfevents.1650513386.DESKTOP-4H81M8H
│
├─prediction_result
│      test_predict_sougou-zhaoshang-fold_1-5_least_data_least_model.csv
│      北交所新股申购业务汇总.csv (以北交所文件为例，模型处理结果)
|
├─pre_trained_model
│  └─NEZHA-Base-WWM
│          bert_config.json
│          model.ckpt-691689.data-00000-of-00001
│          model.ckpt-691689.index
│          model.ckpt-691689.meta
│          vocab.txt
│
└─saved_model
        baseline-sougou-fold-1.h5
        baseline-sougou-zhaoshang-fold-1.h5
        baseline-sougou-zhaoshang-fold-2.h5
        baseline-sougou-zhaoshang-fold-3.h5
        baseline-sougou-zhaoshang-fold-4.h5
        baseline-sougou-zhaoshang-fold-5.h5
```

- `answer_to_question.py`：用于在Sougo数据集上进行预训练
- `answer_to_question_fine_tuning.py`：用于在招商数据集上进行领域适应和微调
- `answer_to_question_inference.py`：用于模型的推断和预测，并在划分好的`test`集上进行验证
- `metric`：生成问题评估指标
- `utils`：数据预处理
- `./data`：数据目录
- `./output`：tensorboard输出目录
- `./prediction_result`：预测结果
- `./pre_trained_model`：NEZHA预训练模型存放目录
- `./saved_model`：生成模型目录
- `file_config`: 存储文件地址，每个需要处理的文件三个地址，一个一行，分别为：源文件地址(`raw_doc`)，文件处理中间地址(`middle`)，模型结果存储地址(`prediction_result`)
- `raw_doc2segs.py`: 将源文件从docx格式进行转换，产生中间结果，用于后续模型生成使用
- `result_generation.py`: 利用被处理后的源文件产生的中间结果文件，进行由答案到问题的生成
- `baseline.sh`: 在配置好`file_config`后，运行该脚本自动产生问答对文件(存储于`file_config`指定的路径上) 



## 使用说明

依次执行以下命令

```shell
python answer_to_question.py
python answer_to_question_fine_tuning.py
python answer_to_question_inference.py
```

评估指标

```
python metric.py
```

由docx文件，进行问答对抽取（需要先设置后`file_config`路径）

```
bash baseline.sh
```

