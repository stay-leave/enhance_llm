from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image


model_name_or_path = "autodl-tmp/vlm_1"  # 

llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16
)


prompt_text = "<image>\n请描述这张图片"


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"prompt: {prompt}")

image_path = "my_code/pictures/1.jpg"
image = Image.open(image_path).resize((336,336))


inputs = llava_processor(text=prompt, images=image, return_tensors="pt")
print("inputs：")
print(inputs)


for tk in inputs.keys():
    inputs[tk] = inputs[tk].to(model.device)

generate_ids = model.generate(**inputs, max_new_tokens=512)
gen_text = llava_processor.batch_decode(
    generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]

print(gen_text)

print("model.config：")
# print(model.config)

