import os
import random
import json
# 加载数据的库
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
# 创建q-p对的库
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
# 导入自定义本地LLM
from qwen import QWEN
# 微调
from llama_index.finetuning import SentenceTransformersFinetuneEngine
import time


# 开始时间
start_time = time.time()


# 源文件 列表
my_list = [i for i in os.listdir('project_2/data') if i.endswith('pdf')]
# 随机抽取70%的数据，作为训练集
random.shuffle(my_list) # 打乱
num_to_sample = int(len(my_list) * 0.7) # 阈值
# 构造本地文件路径
training_set = [f"project_2/data/{file}" for file in my_list[:num_to_sample]] # 训练集文件list
validation_set = [f"project_2/data/{file}" for file in my_list[num_to_sample:]] # 验证集文件list

# 最终形成的训练和验证语料
TRAIN_CORPUS_FPATH = 'project_2/data/corpus/train_corpus.json'
VAL_CORPUS_FPATH = 'project_2/data/corpus/val_corpus.json'

# 读取pdf数据，节点
def load_corpus(files, verbose=False):
    if verbose:
        print(f"正在加载文件 {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"已加载 {len(docs)} 个文档")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"已解析 {len(nodes)} 个节点")

    return nodes

# 保存解析的文本 list
def save_corpus(nodes, path):
    texts = []
    for node in nodes:
        texts.append(node.text)
    
    # 将列表写入 JSON 文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

# 加载数据为节点
train_nodes = load_corpus(training_set, verbose=True)
val_nodes = load_corpus(validation_set, verbose=True)
# 保存文件
save_corpus(train_nodes, TRAIN_CORPUS_FPATH)
save_corpus(val_nodes, VAL_CORPUS_FPATH)
print("加载数据完成！已保存！")


# 自定义提示词模版
template = """\
以下是相关信息。

---------------------
{context_str}
---------------------

请基于上述信息，不使用任何先前知识，
生成以下查询所需的仅包含问题的内容。

你是一名教师/教授，你的任务是为即将到来的 \
测验/考试设置 {num_questions_per_chunk} 个问题。 \
这些问题应在文件中呈现多样性。请将问题限制在 \
提供的上下文信息范围内。"
"""


# 使用LLM生成qa对
train_dataset = generate_qa_embedding_pairs(
    llm=QWEN(model_path="autodl-tmp/Qwen1.5-7B-Chat"), nodes=train_nodes, 
    qa_generate_prompt_tmpl=template, # 自定义模版
    num_questions_per_chunk=2 # 每块文段生成几个问题，默认为2
    )
val_dataset = generate_qa_embedding_pairs(
    llm=QWEN(model_path="autodl-tmp/Qwen1.5-7B-Chat"), nodes=val_nodes,
    qa_generate_prompt_tmpl=template, # 自定义模版
    num_questions_per_chunk=2 # 每块文段生成几个问题，默认为2
    )
# 保存qa对，训练集、验证集
train_dataset.save_json("project_2/data/datasets/train_dataset.json")
val_dataset.save_json("project_2/data/datasets/val_dataset.json")

# [Optional] 加载本地数据集
# train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
# val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
print("LLM生成qa对完成！已保存！")

# 微调配置
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="project_2/bge-large-zh-v1.5", # 模型路径
    model_output_path="project_2/model_outdir/7b-bge", # 微调结构输出路径
    val_dataset=val_dataset, # 验证集（可选）
    batch_size = 10, # 默认为10
    epochs = 2, # 默认为2
    evaluation_steps = 100, # 默认为50步一验证
    )


# 启动
finetune_engine.finetune()

print("微调完成！已保存！")

# 结束时间
end_time = time.time()

# 计算并输出运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.2f} 秒")