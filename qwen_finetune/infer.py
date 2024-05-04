from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
# 修改日志级别
transformers.logging.set_verbosity_error()


device = "cuda" # the device to load the model onto
model_id = "/root/autodl-tmp/Qwen1.5-7B-Chat"
peft_model_id = "/root/autodl-tmp/sft/my-1e-4"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.load_adapter(peft_model_id)


tokenizer = AutoTokenizer.from_pretrained("autodl-tmp/Qwen1.5-7B-Chat")

prompt = "你玩小红书吗？"
messages = [
    {"role": "system", "content": "请扮演一个小红书的用户回答问题。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature = 0.7
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)