import streamlit as st
from openai import OpenAI
import json
import time
import re

# 配置 OpenAI 客户端
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

def make_api_call(messages, max_tokens, temperature):
    """
    发送请求到 OpenAI API，并返回解析后的 JSON 对象。
    包含重试机制，最多尝试3次。
    """
    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None
            )
            content = response.choices[0].message.content.strip()
            # 使用正则表达式提取 JSON 对象
            json_matches = re.findall(r'\{.*?\}', content, re.DOTALL)
            if not json_matches:
                raise ValueError("未找到有效的 JSON 对象。")
            
            # 解析 JSON 对象
            json_str = json_matches[0]
            step_data = json.loads(json_str)
            return step_data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)  # 等待2秒后重试
                continue
            else:
                st.error(f"API 调用失败: {e}")
                st.stop()

def generate_response(prompt, step_container, final_container, progress_bar):
    """
    根据用户的提示，生成逐步推理链，并在前端逐步显示。
    """
    # 系统提示，指导助手逐步推理
    system_prompt = """你是一个专家级的 AI 助手，能够逐步解释你的推理过程。每次只提供一个推理步骤。对于每一步，提供一个描述你在做什么的标题和内容。决定是否需要下一步推理，或者准备给出最终答案。请直接以单一的 JSON 格式回应，包含 'title'、'content' 和 'next_action'（'continue' 或 'final_answer'）键。请勿使用代码块（如 ```json）来包裹 JSON。确保每次回应只包含一个有效的 JSON 对象。
    
    示例有效的 JSON 响应：
    {
        "title": "识别关键信息",
        "content": "要开始解决这个问题，我们需要仔细检查给定的信息，并确定将指导我们解决过程的关键要素。这涉及到...",
        "next_action": "continue"
    }
    """

    # 初始化消息列表
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "谢谢！我将按照指示逐步思考，从分解问题的开始。"}
    ]

    # 存储思考步骤
    steps = []
    step_count = 1  # 索引
    total_thinking_time = 0  # 累加思考时长

    progress_bar.progress(0)
    max_steps = 25

    # 迭代生成推理步骤
    while True:
        start_time = time.time()
        # 发送请求
        step_data = make_api_call(
            messages=messages,
            max_tokens=512,
            temperature=0.3
        )
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        # 添加步骤到思考列表
        step_info = {
            "step": step_count,
            "title": step_data.get('title', '无标题'),
            "content": step_data.get('content', '无内容'),
            "thinking_time": thinking_time
        }
        steps.append(step_info)

        # 更新前端显示
        with step_container.container():
            with st.expander(f"步骤 {step_count}: {step_info['title']}", expanded=True):
                st.write(step_info['content'])
                st.caption(f"耗时：{thinking_time:.2f} 秒")

        # 更新进度条
        progress = min(step_count / max_steps, 1.0)
        progress_bar.progress(progress)

        # 更新消息列表，保持上下文
        messages.append({"role": "assistant", "content": json.dumps(step_data, ensure_ascii=False)})

        # 检查是否达到最终答案
        if step_data.get('next_action') == 'final_answer' or step_count >= max_steps:
            break

        step_count += 1

    # 请求最终答案
    messages.append({"role": "user", "content": "请基于以上推理提供最终答案。"})
    start_time = time.time()
    final_data = make_api_call(
        messages=messages,
        max_tokens=512,
        temperature=0.3
    )
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    # 检查是否有错误
    if final_data.get('title') == '错误':
        final_container.error(f"错误: {final_data.get('content')}")
    else:
        final_step = {
            "step": step_count + 1,
            "title": final_data.get('title', '最终答案'),
            "content": final_data.get('content', '无内容'),
            "thinking_time": thinking_time
        }
        steps.append(final_step)

        # 更新前端显示最终答案
        with final_container.container():
            with st.expander(f"最终答案: {final_step['title']}", expanded=True):
                st.write(final_step['content'])
                st.caption(f"耗时：{thinking_time:.2f} 秒")

    # 更新进度条到100%
    progress_bar.progress(1.0)

    # 显示总思考时间
    final_container.success(f"总思考时间：{total_thinking_time:.2f} 秒")

def main():
    """
    主函数，设置 Streamlit 界面并处理用户输入。
    """
    st.set_page_config(page_title="逐步思维链生成器", layout="wide")
    st.title("OpenAI 逐步思维链生成器")
    st.write("请输入您的查询，系统将逐步展示推理过程，直到得到最终答案。")

    # 用户输入
    prompt = st.text_area("请输入您的查询：", height=100)

    if st.button("生成推理链"):
        if not prompt.strip():
            st.warning("查询不能为空。")
        else:
            # 创建容器用于动态添加步骤
            step_container = st.container()
            final_container = st.container()
            progress_bar = st.progress(0)

            with st.spinner("正在生成推理链，请稍候..."):
                generate_response(prompt, step_container, final_container, progress_bar)

if __name__ == "__main__":
    main()
