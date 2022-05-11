## 环境
```
pytorch>=1.3
transormfers>=4.12.5
datasets>=1.15.1
```

## 数据预处理

```
python -u data_preprocess.py
```

预处理后的数据文件位于`\data`

## 模型训练

运行：

```
python -u run_qa.py config/qa_config.json
```

保存的模型、dev集上的指标计算结果位于`\data`（数据路径下）

## 模型测试

运行：

```
python -u run_qa.py config/qa_eval_config.json
```

指标计算结果位于`\data`

预测结果文件位于`\result`

由于训练数据较少，目前只9:1划分了train、dev，暂时先使用dev作为测试集（因为训练时没有基于dev调参）

 











