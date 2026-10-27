# 本例学习 LangChain 的原生工具调用，以及把工具结果交回模型的基本流程。
# 与 02 的区别：02 让模型生成 <Tool>/<Args> 文本，再用正则手动解析；
# 本例通过LLM提供回答问题所需的工具以及参数，由 LangChain 将调用请求整理到 response.tool_calls。
# 本例流程：声明工具 → 绑定工具 → 模型提出调用请求 → 本地执行 → 回传结果 → 模型回答。

from __future__ import annotations

import os
import sys
from pathlib import Path

# 从 .env 文件加载模型名称、接口地址和密钥等配置。
from dotenv import load_dotenv
# 三种消息角色：System Prompt、User Prompt、ToolMessage。
# 模型回复则由 invoke 返回，通常是 AIMessage对象，后面也需要加入对话历史。
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# tool 是装饰器，用于将 Python 函数包装成 LangChain 可识别的工具对象。
from langchain_core.tools import tool
# ChatOpenAI 也可以连接兼容 OpenAI 接口格式的模型服务。
from langchain_openai import ChatOpenAI

# 没有传入命令行参数时，使用这个问题；它同时涉及两个工具。
DEFAULT_USER_PROMPT = "帮我查一下 Beijing 的天气和时间"
# 一轮指一次模型请求，最多请求 5 次，避免模型反复调用工具而无限循环。
MAX_MODEL_ROUNDS = 5

# System Prompt只说明何时调用工具，如果...请...  不再要求模型手写 <Tool> 等文本标签。
# 工具的名称、说明和参数结构，会通过后面的 bind_tools 提供给模型接口。
SYSTEM_PROMPT = """
你是一个助手。
如果用户询问天气，请调用 get_weather 工具，不要自己编造天气。
如果用户询问时间，请调用 get_time 工具，不要自己编造时间。
本示例的天气和时间都是模拟数据，请在回答中说明。
""".strip()


# @tool 会用函数名作为默认工具名、文档字符串作为说明，并根据参数类型生成参数结构。
# city: str 告诉工具框架：这个工具需要一个名为 city 的字符串参数。
# 装饰后，get_weather 是工具对象，本例通过 get_weather.invoke({...}) 执行它。
@tool
def get_weather(city: str) -> str:
    # 下面的三引号字符串是工具说明，会提供给模型；普通 # 注释不会成为工具说明。
    """Get simulated weather for Beijing, Shanghai, or Hangzhou."""
    # 固定字典模拟天气，没有访问真实天气服务。
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    # get(键, 默认值) 在找不到城市时返回“未知天气”；f-string 拼接返回文本。
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"

# 第二个工具：让模型能区分“查天气”和“查时间”两种请求。
@tool
def get_time(city: str) -> str:
    """Get simulated time for Beijing, Shanghai, or Hangzhou (not real-time)."""
    # 这里返回固定的演示时间，没有读取系统时钟，不能把它当作实时查询结果。
    time = {
        "Beijing": "14:00",
        "Shanghai": "14:00",
        "Hangzhou": "14:00",
    }
    return f"{city} 的当前时间是：{time.get(city, '未知时间')}"

def load_llm() -> ChatOpenAI:
    # __file__ 是本脚本路径；parents[0] 为“1. Toolcall”，parents[1] 为项目目录。
    # / ".env" 拼接出项目根目录的配置文件路径；加载环境变量。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    # 创建模型客户端；真正发送模型请求是在后面的 invoke，而不是这里。
    # os.getenv("名称", 默认值) 优先读取环境变量，没有设置时才采用默认值。
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.5:cloud"),
        base_url=os.getenv("BASE_URL", "http://localhost:11434/v1/"),
        api_key=os.getenv("API_KEY", "ollama"),
        # 较低温度通常让输出更稳定，但不能保证模型一定正确选择工具和参数。
        temperature=0,
    )


def main() -> None:
    # sys.argv[1:] 是脚本名之后的命令行参数；拼接后为空时，使用默认问题。
    # 在项目目录运行示例：
    # python "1. Toolcall/03_langchain_native_toolcall.py" "帮我查一下 Beijing 的天气和时间"
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT

    print("=== 03. LangChain 原生 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    # 第 1 步：收集可用工具，并把它们的描述及参数结构绑定到模型请求中。
    #         两个元素都是经过@tool装饰转换，返回LangChain的工具对象
    tools = [get_weather, get_time]     
    # Python字典推导式，建立tool_name -> tool对象对应关系
    # 相当于得到 { "get_weather": get_weather,"get_time": get_time,}
    tool_map = {t.name: t for t in tools}
    # 创建模型客户端ChatOpenAI，配置其在请求模型时提供的工具
    llm = load_llm().bind_tools(tools)

    # 第 2 步：准备对话历史。列表的先后顺序就是模型接收到的消息顺序。
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    # 第 3 步：外层循环处理多轮模型请求；内层循环执行同一轮中的多个工具请求。
    for round_number in range(1, MAX_MODEL_ROUNDS + 1):
        # llm为前面已绑定工具的模型调用对象，.invoke(message)发送对话消息并等待模型返回AIMessage消息对象
        response = llm.invoke(messages)
        # LLM的回复response加入到messages列表当中,response为AIMessage对象(包含回答文本及工具调用)
        messages.append(response)

        print(f"\n第 {round_number} 轮模型文本输出:")
        print(response.content)     # 只保留内容，丢失消息身份和工具调用信息
        print("\nLangChain 解析出的 tool_calls:")
        print(response.tool_calls)  # 记录tool的调用请求，调用哪个工具及传什么参数。

        # 只要存在解析失败的工具请求，就提示错误并退出当前函数。
        if response.invalid_tool_calls:
            print("\n模型返回了无法解析的工具参数，请检查模型输出后重试。")
            return

        if not response.tool_calls:
            # 当本轮循环不再需要工具调用时，就给出最终回答，结束for循环返回
            # 当本轮模型回答为空或者只是空格、换行等空白字符，LLM未进行回答输出
            if not response.content or (
                isinstance(response.content, str) and not response.content.strip()
            ):
                print("\n模型未返回文本，也未提出工具调用，本次未获得最终回答。")
            else:
            # 模型回答不为空
                print("\n最终回答:")
                print(response.content)
            return

        # response.tool_calls单项示意：{"name": "get_time", "args": {"city": "Beijing"}, "id": "call_123"}。
        # name 选择工具，args 提供参数，id 把结果与对应请求关联起来。
        for tool_call in response.tool_calls:
            print("\n准备执行工具:")
            print(f"tool_name = {tool_call['name']}")
            print(f"tool_args = {tool_call['args']}")

            # 根据tool_name取出 tool_map 对象赋值给 selected_tool，得到LangChain的BaseTool对象。
            selected_tool = tool_map.get(tool_call["name"])
            status = "success"
            if selected_tool is None:
                result = f"未知工具：{tool_call['name']}。可用工具：{', '.join(tool_map)}"
                status = "error"
            else:
                try:
                    # invoke校验参数并执行选中的工具, BaseTool.invoke(tool_args)传回工具返回结果
                    result = selected_tool.invoke(tool_call["args"])
                except Exception as exc:
                    # 把参数错误、执行失败作为工具结果回传，模型可在下一轮尝试纠正。
                    result = f"工具执行失败（{type(exc).__name__}）：{exc}"
                    status = "error"

            print("\n工具执行结果:")
            print(result)
            # ToolMessage 是 LangChain提供的一个消息类，用来表示“工具执行后返回的结果”
            messages.append(
                ToolMessage(
                    content=result,
                    name=tool_call["name"],
                    # 每个请求对应一条结果，使用原请求的 id，不要自行生成新的 id。
                    tool_call_id=tool_call["id"],
                    status=status,
                )
            )

        # 第 4 步：本轮全部工具执行完，外层循环将完整历史再次发送给模型。
        # 正确顺序：系统 → 用户 → 模型请求（一次）→ 多个工具结果 → 下一轮模型回复。

    print(f"\n已达到 {MAX_MODEL_ROUNDS} 轮模型请求上限，停止运行；尚未获得最终回答。")


# 直接运行这个文件才执行 main；作为模块导入时，不会自动发起模型请求。
if __name__ == "__main__":
    main()
