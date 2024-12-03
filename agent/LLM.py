from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI




class InternLM2Chat():
    def __init__(self, ) -> None:
        super().__init__()
        self.load_model()
        self.messages = []
        
    def load_model(self):
        print('================ Loading model ================')
        self.client = OpenAI(api_key="sk-bf21fcc37a07487ea72fb7f5aa82ad18", base_url="https://api.deepseek.com")
        print('================ Model loaded ================')

    def chat(self) -> str:
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=self.messages,
            temperature = 0.3,
            stream=False
        )

        response = response.choices[0].message
        self.messages.append({"role": response.role, "content": response.content})

        return response.content
    

if __name__ == '__main__':

    model = InternLM2Chat()
    model.messages = [
            {"role": "user", "content": "特朗普赢得了最新的美国大选了吗？"}
        ]
    print(model.chat())
    

