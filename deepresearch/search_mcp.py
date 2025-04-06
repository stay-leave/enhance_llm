from mcp.server.fastmcp import FastMCP
import requests
import pandas as pd
import logging
from zhipuai import ZhipuAI
from typing import List, Dict
import yaml
import json
import infinity_embedded as infinity
from transformers import AutoModel, AutoModelForSequenceClassification
import torch
from utils import *
import time

def get_clear_json(text):
    if '```json' not in text:
        return text
    
    return text.split('```json')[1].split('```')[0]

# 读取yaml文件
def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = yaml.safe_load(file)
        return prompts


mcp = FastMCP("local_web_search")

# 大模型，示例化客户端
base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
api_key = 'aaa'
model_name = 'glm-4v-flash'
client = ZhipuAI(
            api_key=api_key
        )

# 知识库加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 向量模型初始化
model = AutoModel.from_pretrained('autodl-tmp/models/jina-clip-v2', trust_remote_code=True).to(device).eval()
# 重排模型初始化
rerank_model = AutoModelForSequenceClassification.from_pretrained('autodl-tmp/models/jina-reranker-v2-base-multilingual',trust_remote_code=True).to(device).eval()
# 连接数据库服务端，本地文件夹，必须是绝对路径
infinity_object = infinity.connect("/root/RAG/database", config_path="/root/RAG/infinity_config/infinity_conf.toml")
db_object = infinity_object.get_database("paper")

# 提示词
prompts = load_prompts("deepresearch/prompts.yaml")


# 改写查询
def rewrite_query(query):
    
    prompt = prompts["QUERY_REWRITE"]
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"用户查询: {query}"}
        ]
    )

    text = response.choices[0].message.content
    dict_str = get_clear_json(text)
    res = json.loads(dict_str)["enhanced_query"]

    return res


def if_useful(query: str, page_text: str) -> str:
    prompt = """你是一个严格的研究评估员。根据用户的查询和网页内容，判断该网页是否包含与解决查询相关且有用的信息。
    仅返回一个单词：'是' 如果网页有用，或 '否' 如果没用。不要包含任何额外文本。"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个严格且简洁的研究相关性评估员。"},
            {"role": "user", "content": f"用户查询: {query}\n\n网页内容（前200字符）:\n{page_text[:200]}\n\n{prompt}"}
        ]
    )
    
    response = response.choices[0].message.content
    
    if response:
        answer = response.strip()
        if answer in ["是", "否"]:
            return answer
        else:
            # 备选方案：尝试从响应中提取“是”或“否”。
            if "是" in answer:
                return "是"
            elif "否" in answer:
                return "否"
    return "否"


def extract_relevant_context(query: str, search_query: str, page_text: str) -> str:
    prompt = """你是一个专业的信息提取员。根据用户的查询、此页面的搜索查询以及网页内容，提取所有与回答用户查询相关的信息。
    仅返回相关的上下文作为纯文本。"""
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个提取和总结相关信息的专家。"},
            {"role": "user", "content": f"用户查询: {query}\n搜索查询: {search_query}\n\n网页内容（前20000字符）:\n{page_text[:20000]}\n\n{prompt}"}
        ]
    )
    
    response = response.choices[0].message.content
    if response:
        return response.strip()
    return ""


# 网页搜索
def web_search(query: str) -> List[str]:
    
    url = "https://s.jina.ai/"
    params = {"q": query}
    headers = {
        "Authorization": "Bearer ",
        "X-Respond-With": "no-content",
        'Accept': 'application/json'
    }
    response = requests.get(url, headers=headers, params=params)

    res_list = json.loads(response.text)["data"]

    return res_list

# 读取网页内容
def fetch_webpage_text(url: str) -> str:
    JINA_BASE_URL = 'https://r.jina.ai/'
    full_url = f"{JINA_BASE_URL}{url}"
    
    resp = requests.get(full_url, timeout=50)
    if resp.status_code == 200:
        return resp.text
    return ""

# 处理单个url
def process_link(link: str, query: str, search_query: str) -> str:
    # 尝试获取网页文本内容
    page_text = fetch_webpage_text(link)
    time.sleep(1)
    if not page_text:
        return None
    # 判断有用性
    usefulness = if_useful(query, page_text)
    # 提取相关文本
    if usefulness == "是":
        context = extract_relevant_context(query, search_query, page_text)
        if context:
            return context
    return None


# 判断是否继续搜索
def get_new_search_query(user_query: str, previous_search_queries: List[str], all_contexts: List[str]) -> str:
    """
    根据原始查询、之前的搜索查询和提取的上下文，判断是否需要进一步研究。
    如果需要进一步研究，返回一个新的搜索查询（字符串形式）。
    如果不需要进一步研究，返回空字符串。
    """
    context_combined = "\n".join(all_contexts)
    
    prompt = """你是一个分析性的研究助手。基于以下信息，判断是否需要进一步研究：
    1. 原始查询：用户的初始问题或需求。
    2. 之前的搜索查询：已经执行过的搜索关键词。
    3. 提取的相关上下文：从网页中提取的相关信息。

    任务要求：
    - 如果需要进一步研究，请提供一个新的搜索查询（字符串形式）。
    - 如果认为不需要进一步研究，请返回空字符串。
    - 输出仅限一个字符串，不要包含任何额外文本。"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个提取和总结相关信息的专家，能够精准地判断是否需要进一步研究。"},
            {"role": "user", "content": f"""
            用户查询: {user_query}
            之前的搜索查询: {previous_search_queries}

            提取的相关上下文:
            {context_combined}

            {prompt}
            """}
        ]
    )
    
    # 获取模型的响应内容
    response_content = response.choices[0].message.content.strip()
    
    # 判断响应内容是否为空
    if not response_content or response_content == "":
        return ""
    
    # 返回新的搜索查询
    return response_content

# 图像描述
def get_images_description(image_url: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "用一句话描述图片的内容"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content


# 完成网络搜索的代码，纯文本

def search(query: str) -> str:
    """互联网搜索"""
    iteration_limit = 3  # 最大迭代次数
    iteration = 0
    aggregated_contexts = []  # 聚合的搜索结果
    all_search_queries = []   # 记录所有搜索查询
    
    # 初始查询改写
    current_query = rewrite_query(query)
    all_search_queries.append(current_query)

    while iteration < iteration_limit:
        print(f"\n=== 第 {iteration + 1} 次迭代 ===")
        print(f"用户原始查询：{query} | 当前搜索查询：{current_query}")
        
        # 执行搜索并处理结果
        search_results = web_search(current_query)
        links = [res["url"] for res in search_results] if search_results else []
        
        # 并行处理链接并提取上下文
        iteration_contexts = []
        for link in links:
            context = process_link(link, query, current_query)
            if context:
                iteration_contexts.append(context)
        
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            print("本次迭代未找到有用信息。")
        
        # 生成新查询
        new_query = get_new_search_query(
            user_query=query,
            previous_search_queries=all_search_queries,
            all_contexts=aggregated_contexts
        )
        
        # 判断是否需要继续迭代
        if not new_query:
            print("无需进一步研究。")
            break
        elif new_query in all_search_queries:
            print(f"查询 {new_query} 已执行过，停止迭代。")
            break
        
        # 更新查询和记录
        current_query = new_query
        all_search_queries.append(current_query)
        iteration += 1

    return '\n\n'.join(aggregated_contexts)


# 本地知识库检索
@mcp.tool()
def retrieve(query: str) -> str:
    """本地知识库检索"""
    iteration_limit = 3  # 最大迭代次数
    iteration = 0
    aggregated_contexts = []  # 聚合的检索结果
    all_search_queries = []   # 所有查询记录
    
    # 初始查询改写
    current_query = rewrite_query(query)
    all_search_queries.append(current_query)

    while iteration < iteration_limit:
        print(f"\n=== 第 {iteration + 1} 次迭代 ===")
        print(f"用户原始查询：{query} | 当前检索查询：{current_query}")
        
        # 执行本地知识库混合检索
        try:
            chunk_df, image_df, _ = chunk_hybid_search(db_object, model, current_query)
            iteration_contexts = chunk_df["restored_content"].tolist()
        except Exception as e:
            print(f"检索失败：{e}")
            break
        
        # 处理检索结果
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            print("本次迭代未找到有用信息。")
        
        # 生成新查询
        new_query = get_new_search_query(
            user_query=query,
            previous_search_queries=all_search_queries,
            all_contexts=aggregated_contexts
        )
        
        # 终止条件判断
        if not new_query:
            print("无需进一步研究。")
            break
        elif new_query in all_search_queries:
            print(f"查询 {new_query} 已执行过，停止迭代。")
            break
        
        # 更新查询和记录
        current_query = new_query
        all_search_queries.append(current_query)
        iteration += 1

    return '\n\n'.join(aggregated_contexts)


# 混合检索

def muti_search(query: str) -> str:
    """互联网和本地知识库混合搜索"""
    iteration_limit = 3  # 最大迭代次数
    iteration = 0
    aggregated_contexts = []  # 聚合的搜索结果
    all_search_queries = []   # 所有查询记录
    
    # 初始查询改写
    current_query = rewrite_query(query)
    all_search_queries.append(current_query)

    while iteration < iteration_limit:
        print(f"\n=== 第 {iteration + 1} 次迭代 ===")
        print(f"用户原始查询：{query} | 当前搜索查询：{current_query}")
        
        # 互联网搜索
        try:
            web_results = web_search(current_query)
            web_links = [res["url"] for res in web_results] if web_results else []
        except Exception as e:
            print(f"互联网搜索失败：{e}")
            web_links = []
        
        # 处理互联网搜索结果
        web_contexts = []
        for link in web_links:
            context = process_link(link, query, current_query)
            if context:
                web_contexts.append(context)
        
        # 本地知识库检索
        try:
            chunk_df, image_df, _ = chunk_hybid_search(db_object, model, current_query)
            local_contexts = chunk_df["restored_content"].tolist()
        except Exception as e:
            print(f"本地检索失败：{e}")
            local_contexts = []
        
        # 合并结果
        iteration_contexts = web_contexts + local_contexts
        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            print("本次迭代未找到有用信息。")
        
        # 生成新查询
        new_query = get_new_search_query(
            user_query=query,
            previous_search_queries=all_search_queries,
            all_contexts=aggregated_contexts
        )
        
        # 终止条件判断
        if not new_query:
            print("无需进一步研究。")
            break
        elif new_query in all_search_queries:
            print(f"查询 {new_query} 已执行过，停止迭代。")
            break
        
        # 更新查询和记录
        current_query = new_query
        all_search_queries.append(current_query)
        iteration += 1

    return '\n\n'.join(aggregated_contexts)



if __name__ == "__main__":
    
    mcp.run()
    

    
    

    
