from typing import Dict, List, Optional, Tuple, Union
import json5
from LLM import InternLM2Chat
from tool import Tools

# 工具描述
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
# 提示词
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""


class Agent:
    def __init__(self) -> None:
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = InternLM2Chat()
        self.initialize_messages()

    def build_system_input(self):
        # 系统提示词
        tool_descs, tool_names = [], []  # 工具描述，工具名称
        # 遍历工具信息
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))  # 将工具信息转换为描述
            tool_names.append(tool['name_for_model'])  # 添加工具名
        # list转str
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        # 将工具描述和工具名输入系统提示词
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def initialize_messages(self):
        # 初始化消息历史，仅添加系统提示一次
        self.model.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

    def parse_latest_plugin_call(self, text):
        # 解析最新的tool call
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:
            if k < j:
                text = text.rstrip() + '\nObservation:'
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:'):j].strip()
            plugin_args = text[j + len('\nAction Input:'):k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        # 进行工具使用
        plugin_args = json5.loads(plugin_args)  # 解析输入为json
        if plugin_name == 'google_search':
            res = self.tool.google_search(**plugin_args)
            return '\nObservation:' + res  # 直接加结果，是action后的结果

    def text_completion(self, text):
        # 大模型调用
        user_message = {"role": "user", "content": f"\nQuestion:{text}"}
        self.model.messages.append(user_message)
        
        # 第一次调用模型
        response = self.model.chat()
        
        # 解析思考的结果
        plugin_name, plugin_args, updated_response = self.parse_latest_plugin_call(response)
        
        # 如果要用工具，就进行call
        if plugin_name:
            observation = self.call_plugin(plugin_name, plugin_args)
            # 将Observation添加到消息历史
            self.model.messages.append({"role": "assistant", "content": updated_response + observation})
            
            # 第二次调用模型，继续推理
            final_response = self.model.chat()
            return final_response
        else:
            # 如果模型直接给出Final Answer，没有调用工具
            return response


if __name__ == '__main__':
    agent = Agent()
    prompt = "特朗普赢得了最新的美国大选了吗？"
    res = agent.text_completion(prompt)
    print(res)
    print(agent.model.messages)
