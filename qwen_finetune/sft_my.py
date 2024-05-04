from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig, TrainerCallback
import logging
import torch
from peft import LoraConfig, TaskType, get_peft_model
import wandb

wandb.init(
    project="qwen_lora",  # 设置项目名称
    name="1e-4"  # 设置运行名称
)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('project_3/fine-tune/output_sft_my.log'),
        logging.StreamHandler()
    ]
)
# 获取对象
logger = logging.getLogger(__name__)


df = pd.read_json('project_3/fine-tune/data/COIG-CQIA-full.json')
df = df[['instruction', 'input', 'output']]
# 直接从 Pandas 数据框创建 Dataset 对象
ds = Dataset.from_pandas(df)
# 记录日志信息
logger.info(f"Dataset ds[:2]: {ds[:2]}")


# 模型路径
model_path = '/root/autodl-tmp/Qwen1.5-7B-Chat'
# 分词器载入
# "Fast" Tokenizer 是由 tokenizers 库提供的 Rust 实现，通常比 Python 实现更快，并且支持更高效的分词操作。
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
logger.info(f"tokenizer: {tokenizer}")

# 每条样本的处理-文本转token
def process_func(example):
    MAX_LENGTH = 1024    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    # 模型输入，跟传统的nlp一样
    input_ids, attention_mask, labels = [], [], []
    # 根据qwen1.5的prompt设计，参考llama-factory
    system = '现在你要扮演一个中文社交媒体的活跃用户，热心回答别人的问题。' # "You are a helpful assistant.",
    system = f"<|im_start|>system\n现在你要扮演一个中文社交媒体的活跃用户，热心回答别人的问题。<|im_end|>\n"
    instruction = f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n" # 将instrction和input合并
    response = f"{example['output']}"
    # 分词，prompt = system+instruction
    prompt = tokenizer(system + instruction, add_special_tokens=False) # 不在开头加 special_tokens
    response = tokenizer(response, add_special_tokens=False) # 不在开头加 special_tokens
    # input_ids，填充长度，对齐
    # attention_mask 中的 1 用于指示模型需要关注输入序列的哪些部分，而 labels 中的 -100 用于指示训练期间应忽略哪些部分。它们一起确保了 NLP 模型在训练期间专注于实际内容，并忽略填充或其他不相关部分。
    input_ids = prompt['input_ids'] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = prompt["attention_mask"] + response["attention_mask"] + [1] # 加1表示关注
    labels = [-100] * len(prompt["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] # 要预测的其实是回复，所以要加-100忽略promt
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 将数据集转换为输入的数据集
tokenized_id = ds.map(
    process_func, # `process_func` 是一个函数，用于将每个示例转换为模型可接受的格式。
    remove_columns=ds.column_names, # 删除数据集中的原始列，以避免与转换后的格式冲突。
    num_proc=20 # 使用 20 个 CPU 核心并行处理数据集，以加速转换过程。
)
logger.info(f"tokenizer.decode(tokenized_id[0]['input_ids']): {tokenizer.decode(tokenized_id[0]['input_ids'])}")

# 将数据集划分为训练集和测试集 9:1
my_datasets = tokenized_id.train_test_split(
    test_size=0.1 # 将 10% 的数据划分为测试集，剩余的 90% 用于训练。
)



device = "cuda"
# 模型加载
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
logger.info(f"model.dtype: {model.dtype}")
logger.info(f"model: {model}")

# Lora 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 指定任务类型为 Causal Language Model（因果语言模型），表示模型用于生成文本。
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 指定应用 LoRA 的目标模块，如投影层和其他重要模块。
    inference_mode=False, # 指定为训练模式，关闭推理模式，以进行模型训练。
    r=8, # LoRA 的秩（Rank），表示降维矩阵的秩。较高的秩可以提高表示能力。
    lora_alpha=32, # LoRA 的 alpha 值，用于缩放权重矩阵。较大的 alpha 值表示更强的缩放效果。
    lora_dropout=0.1 # Dropout 比例，用于防止过拟合。
)
logger.info(f"config: {config}") # 记录配置日志。

# 定义 LoRA 模型
lora_model = get_peft_model(model, config) # 基于原模型和 LoRA 配置创建新的模型。
logger.info(f"lora_model config: {config}") # 记录模型配置日志。
logger.info(f"model.print_trainable_parameters(): {lora_model.print_trainable_parameters()}") # 打印可训练参数数量。


# 定期清理CUDA缓存
class EmptyCudaCacheCallback(TrainerCallback):
    log_cnt = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.log_cnt += 1
        if self.log_cnt % 5 == 0:
            torch.cuda.empty_cache()
empty_cuda_cahce = EmptyCudaCacheCallback()

# 定义训练参数
args = TrainingArguments(
    output_dir='/root/autodl-tmp/sft/my-1e-4', # 模型训练输出的目录，包括保存模型和其他训练输出。
    per_device_train_batch_size=1, # 每个设备上的训练批量大小。注意在 GPU 上每个批次中训练的数据量。
    gradient_accumulation_steps=8, # 梯度累积步骤，将 8 个批次的梯度累积后再更新模型参数。
    num_train_epochs=5, # 总共训练 5 个 epochs。
    weight_decay=0.1, # 权重衰减（L2 正则化），用于防止过拟合。
    warmup_steps=90, # 热身步骤数，在此期间学习率从 0 逐渐增加到预定值。
    learning_rate=1e-4, # 初始学习率，用于梯度下降。
    ddp_find_unused_parameters=False, # 分布式训练参数，指示是否查找未使用的参数。
    evaluation_strategy="steps", # 每隔一定步骤评估模型。可选值有 "steps" 和 "epoch"。
    eval_steps=100, # 每隔 100 步评估模型性能。
    save_steps=200, # 每隔 200 步保存一次模型检查点。
    save_total_limit=5, # 保持的最大模型检查点数量，以免占用过多磁盘空间。
    report_to="wandb", # 报告训练日志到 Weights & Biases。可选"tensorboard"、“wandb”
    optim="adamw_torch", # 使用 AdamW 优化器。
    remove_unused_columns=False, # 保留所有数据集中的原始列。
    lr_scheduler_type="cosine", # 使用余弦退火学习率策略。
    bf16=True, # 使用 bfloat16 混合精度训练，以减少显存占用。
    logging_steps=10, # 每隔 10 步打印一次训练日志。
    log_level="info", # 设置日志输出级别为 "info"。
    logging_first_step=True, # 在第一次训练步骤后输出日志。
    seed=42 # 随机数种子，用于确保训练结果可复现。
)

# 实例化 Trainer 类
trainer = Trainer(
    model=lora_model, # 要训练的模型。
    tokenizer=tokenizer, # 分词器，用于将文本转换为模型输入。
    args=args, # 训练参数。
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 数据整理器，用于处理训练输入。
    train_dataset=my_datasets["train"], # 训练数据集。
    eval_dataset=my_datasets["test"], # 测试数据集。
    callbacks=[empty_cuda_cahce], # 定期清理 CUDA 缓存的回调函数。
)


# 训练
trainer.train()
# 评估损失
eval_results = trainer.evaluate()
# 保存模型
lora_model.save_pretrained(args.output_dir)