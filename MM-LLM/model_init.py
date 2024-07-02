from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor
import torch
from transformers import (
    LlavaForConditionalGeneration,
    LlavaConfig
)

# 模型权重路径
modify_qwen_tokenizer_dir = "autodl-tmp/Qwen2-1.5B-Instruct"
clip_model_name_or_path = (
    "autodl-tmp/clip-vit-large-patch14-336"
)

# 加载qwen2
qwen_tokenizer = AutoTokenizer.from_pretrained(modify_qwen_tokenizer_dir)
qwen_model = AutoModelForCausalLM.from_pretrained(
                                            modify_qwen_tokenizer_dir, 
                                            device_map='cuda:0', 
                                            torch_dtype=torch.bfloat16
                                            )


# 加载clip
clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map="cuda:0")
processor = AutoProcessor.from_pretrained(clip_model_name_or_path)

# 将clip模型和llm_model模型的config拿出来，初始化一个llava model
# Initializing a CLIP-vision config
vision_config = clip_model.vision_model.config
# Initializing a Llama config
text_config = qwen_model.config
# Initializing a Llava llava-1.5-7b style configuration
configuration = LlavaConfig(vision_config, text_config)
# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration(configuration)


# 检查配置
print(model.vision_tower.vision_model.embeddings)
print()
print(clip_model.vision_model)
print(model)
print()

# 权重复制
model.vision_tower.vision_model = clip_model.vision_model
model.language_model = qwen_model

print(model.config.pad_token_id) # None
model.config.pad_token_id = qwen_tokenizer.pad_token_id
print(model.config.pad_token_id) # 151643

print(model.config.image_token_index) # 32000
model.config.image_token_index = qwen_tokenizer.encode("<image>")[0]
print(model.config.image_token_index) # 151646


# 保存模型
model.save_pretrained("autodl-tmp/vlm_1")
qwen_tokenizer.save_pretrained("autodl-tmp/vlm_1")
processor.save_pretrained("autodl-tmp/processor")
