

# rjl代码搜索

## 关于unixcoder（预训练生成embedding）
对于代码搜索，unixcoder是用于：
- 计算code和NL的embeddings
- 计算二者的相似度
- 排序，输出搜索结果

## 关于code-search和run.py
以AdvTest数据集为例:
- Zero-shot是直接用AdvTest的测试集，评估预训练模型的MRR
- 微调则是用AdvTest的训练集和验证集进行微调，然后用测试集评估MRR

也就是说，这个run.py是用来得出论文中实验结果的，开销大，不要在自己电脑上随便尝试

## 但是可以用run.py的逻辑功能，完成智慧编程的代码搜索功能部署
具体做法:（难度较大，要读懂后端代码，前端也要适配好）
- 前端提供自然语言查询的输入
- 该查询传给这个项目生成embedding
- 查询和本地代码库/数据集中的embedding计算相似度并排序
- 返回前端，用表格之类的形式展示给用户