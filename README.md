#### KG Evaluation

---
- 本项目是通过BERT模型，对知识图谱进行评估训练
- head和tail的token_type为0，relation的token_type为1
- 通过dataset文件对数据进行处理
- 本项目参考了项目：https://github.com/yao8839836/kg-bert

---
#### 项目介绍
- data文件夹下是所有的数据集
- out_bert是训练好的模型，每一轮训练会生成一个文件，用来存储模型和词向量
- src存储的是模型的各种文件，用来定义参数、处理数据、定义训练过程、定义模型
- processor中存储的是项目的训练评估文件

---
#### 项目运行
- 项目训练运行processor/train.py
- 项目评估运行processor/eval.py
