# 自定义评测配置文件

from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.ceval.ceval_gen import ceval_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets



# 将需要评测的数据集拼接成 datasets 字段
datasets = [*mmlu_datasets, *ceval_datasets, *gsm8k_datasets]

# 使用 HuggingFaceCausalLM 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 HuggingFaceCausalLM 的初始化参数
        path='/root/autodl-tmp/Qwen1.5-7B-Chat',
        model_kwargs=dict(torch_dtype="auto",device_map='auto'),
        tokenizer_path='/root/autodl-tmp/Qwen1.5-7B-Chat',
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # 以下参数为各类模型都必须设定的参数，非 HuggingFaceCausalLM 的初始化参数
        abbr='Qwen-7b',            # 模型简称，用于结果展示
        max_out_len=100,            # 最长生成 token 数
        batch_size=16,              # 批次大小
        run_cfg=dict(num_gpus=1, num_procs=16),   # 运行配置，用于指定资源需求
    )
]