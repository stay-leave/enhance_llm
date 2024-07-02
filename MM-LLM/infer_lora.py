from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from peft import peft_model,PeftModel

raw_model_name_or_path = "autodl-tmp/vlm_1"
peft_model_name_or_path = "autodl-tmp/output"
model = LlavaForConditionalGeneration.from_pretrained(raw_model_name_or_path,device_map="cuda:0", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, peft_model_name_or_path, adapter_name="peft_v1")
processor = AutoProcessor.from_pretrained(raw_model_name_or_path)
model.eval()


llava_processor = LlavaProcessor.from_pretrained(raw_model_name_or_path)


prompt_text = "<image>\n请描述这张图片"


messages = [
    {"role": "system", "content": "你是一个强大的人工智能助手。"},
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



