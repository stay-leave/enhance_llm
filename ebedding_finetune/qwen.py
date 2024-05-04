from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

class QWEN(CustomLLM):
    context_window: int = 32768
    num_output: int = 4096
    model_name: str = "qwen1.5-1.8B"
    dummy_response: str = "My response"
    model_path: str = 'qwen1.5-1.8B'

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)

if __name__ == "__main__":
    # define our LLM
    Settings.llm = QWEN()

    # define embed model
    Settings.embed_model = "local:project_2/bge-large-zh-v1.5"


    # Load the your data
    documents = SimpleDirectoryReader("project_2/data").load_data()
    index = SummaryIndex.from_documents(documents)

    # Query and print response
    query_engine = index.as_query_engine()
    response = query_engine.query("<query_text>")
    print(response)