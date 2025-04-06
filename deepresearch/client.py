from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional
from contextlib import AsyncExitStack
import json
import asyncio
import yaml
from zhipuai import ZhipuAI

base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
api_key = 'aaa'
model_name = 'glm-4v-flash'

# 读取yaml文件
def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)
        return prompts
# 提示词加载
prompts = load_prompts("deepresearch/prompts.yaml")

# 获取json格式文本
def get_clear_json(text):
    if '```json' not in text:
        return 0, text
    return 1, text.split('```json')[1].split('```')[0]


# 客户端
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = ZhipuAI(
            api_key=api_key
        )
    
    # 启动本地工具服务器（如 search_mcp.py），并通过标准输入/输出与其通信
    async def connect_to_server(self, server_script_path: str):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # 列出可用工具
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to server with tools: {[tool.name for tool in tools]}")
    
    # 进入查询处理流程
    async def process_query(self, query: str) -> str:
        """使用 LLM 和 MCP 服务器提供的工具处理查询"""
        # 列出所有的工具
        response = await self.session.list_tools()
        
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        print(f'available_tools:\n{available_tools}')
        
        # 初始规划，选择工具，返回工具名称和参数
        messages = [
            {
                "role": "system",
                "content": prompts["SYSTEM_PROMPT"] + str(available_tools)
            },
            {
                "role": "user",
                "content": query
            }
        ]
        response = self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        
        message = response.choices[0].message
        print(f'llm_output(tool call)：{message.content}') # 这一步直接给我返回大模型的结果了，无语
        
        # 确定好工具，进行循环
        results = [] # 工具返回的结果聚合
        while True:
            
            flag, json_text = get_clear_json(message.content)# 根据是否能解析出json，执行不同的方法
            
            if flag == 0: # 没有生成json格式，直接用大模型生成回复
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": query}]
                )
                return response.choices[0].message.content
            
            # 成功生成json，解析出工具名和参数
            json_text = json.loads(json_text)
            tool_name = json_text['name']
            tool_args = json_text['params']
            # 执行工具函数，获得返回值
            result = await self.session.call_tool(tool_name, tool_args)
            print(f'tool name: \n{tool_name}\ntool call result: \n{result}')
            results.append(result.content[0].text)
            
            # 把返回工具回复加入历史消息列表
            messages.append({
                "role": "assistant",
                "content": message.content
            })
            messages.append({
                "role": "user",
                "content": f'工具调用结果如下：{result}'
            })
            
            # 在工具调用完成后，由大模型决定是否结束检索，还是继续检索
            messages.append({
                "role": "user",
                "content": prompts["NEXT_STEP_PROMPT"].format(query)
            })
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            
            message = response.choices[0].message
            print(f'llm_output：\n{message.content}')
            # 检查是否该结束
            if 'finish' in message.content:
                break
            
            # 继续检索，就把大模型的回复加入历史消息，继续循环
            messages.append({
                "role": "assistant",
                "content": message.content
            })
        
        # 循环终止后，进入报告撰写阶段
        messages.append({
                "role": "user",
                "content": prompts["FINISH_GENETATE"].format('\n\n'.join(results), query)
                })
        
        response = self.client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        # 返回报告内容
        message = response.choices[0].message.content
        return message
    

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")
                


async def main():
   
    client = MCPClient()
    
    await client.connect_to_server('deepresearch/search_mcp.py')
    
    await client.chat_loop()
    
    

if __name__ == "__main__":
    asyncio.run(main())