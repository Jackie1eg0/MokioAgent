# 学习目标：让真实LLM通过“约定格式的文本”提出工具调用请求。
# 与 01的区别：01手写模型输出；本例会向LLM服务发送消息，获取真实回复。
# 整体流程：用户问题 → Prompt + 问题 发给LLM → 解析LLM回复 → 执行本地工具。
# 这里没有使用模型 API 原生的 tool_calls 字段，而是用提示词约定文本格式。

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# 以下来自第三方依赖：python-dotenv、langchain-core、langchain-openai。
# load_dotenv：把 .env 文件中的配置加载到进程的环境变量中，通常是APIKey。
from dotenv import load_dotenv
# SystemMessage 表达系统规则，HumanMessage 表达用户的具体请求。
from langchain_core.messages import HumanMessage, SystemMessage
# ChatOpenAI 是模型客户端，也可连接兼容 OpenAI 接口格式的其他模型服务。
from langchain_openai import ChatOpenAI

# 运行脚本时没有提供问题，就使用这句默认问题。
DEFAULT_USER_PROMPT = "帮我查一下 Beijing 的天气"

# System Prompt：告诉模型“有哪些工具、什么时候使用、按什么格式提出请求”。
# 例如格式可规定：<Tool> 中放工具名，<Args> 中放 JSON 参数；Beijing 是参数格式示例。
# 这些标签是我们自己约定的普通文本，为了隔离LLM回答的不同类别内容。
# 三引号用于多行字符串，末尾的 strip() 去掉整个字符串首尾的空白。
SYSTEM_PROMPT = """
你是一个助手。
系统里有一个工具叫 get_weather，用于查询天气。

当用户询问天气时，不要直接回答。
你必须严格输出下面的格式，不要输出额外内容：

<Tool>get_weather</Tool>
<Args>{"city":"Beijing"}</Args>

如果用户不需要查询天气，就直接输出普通文本。
""".strip()


def get_weather(city: str) -> str:
    # tool就是可被程序调用的函数；city: str 和 -> str 说明输入和输出是字符串。
    # 这里仍使用固定字典模拟天气，没有调用真实天气 API。
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    # 按城市名查字典，找不到时返回“未知天气”；f-string 将结果填入文本。
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"


def load_llm() -> ChatOpenAI:
    # 函数创建并返回模型客户端。
    # 定位项目根目录下的.env文件，将其中配置加载到当前Python进程环境变量中，为后续os.getenv()读取
    # __file__为当前.py文件路径，Path(...).resolve()得到解析后绝对路径，.parents[1]取得上两层目录([0]为上层目录)
    # /“.env”拼接得到.env路径，读取配置加载到环境变量。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        # os.getenv("变量名", 默认值)：优先读环境变量MODEL，不存在时使用默认值。
        # MODEL 指定模型名称，服务端需要能提供这个模型。
        model=os.getenv("MODEL", "qwen3.5:cloud"),
        # BASE_URL 指定模型服务地址；默认连接本机 11434 端口的兼容接口。
        base_url=os.getenv("BASE_URL", "http://localhost:11434/v1/"),
        # API_KEY 是服务的认证配置；默认字符串 "ollama" 不是真实云服务密钥。
        # 如果接入需要认证的服务，应在环境变量或 .env 中填写该服务的有效密钥。
        api_key=os.getenv("API_KEY", "ollama"),
        # temperature 控制生成的随机程度；设为 0 通常更稳定，但不保证格式正确。
        temperature=0,
    )


# 对LLM生成的Tool请求调用进行解析，返回指定形式的调用回答
# dict[str, object] 表示键是字符串、值可以是不同类型；| None 表示也可能为空。
def parse_tool_call(text: str) -> dict[str, object] | None:
    # re.search 用正则表达式寻找标签，成功返回匹配对象，失败返回 None。
    # r"..." 是原始字符串写法，常用于正则表达式。
    # (.*?) 捕获标签内的内容：. 匹配字符，* 表示零次或多次，? 表示尽量少匹配。
    tool_match = re.search(r"<Tool>(.*?)</Tool>", text, re.DOTALL)
    args_match = re.search(r"<Args>(.*?)</Args>", text, re.DOTALL)

    if not tool_match:
        # 没找到工具标签就不执行工具，main 会结束本次流程。
        # 这也可能是模型输出格式错误，并不能完全证明模型原本没有调用工具的意图。
        return None

    # group(1) 取第一个括号捕获的内容，也就是 <Args> 和 </Args> 之间的文本。
    # json.loads 将 JSON 文本 '{"city":"Beijing"}' 转成 Python 字典。
    # “A if 条件 else B”是条件表达式：找到参数标签就解析，否则使用空字典。
    # 注意：JSON 格式不合法会抛出异常；本例没有验证解析结果是否为字典、是否包含 city 等必要参数，所以格式正确的 JSON 也不一定能用于调用工具。
    args = json.loads(args_match.group(1)) if args_match else {}
    # 统一返回结构，例如：{"tool": "get_weather", "args": {"city": "Beijing"}}。
    return {
        "tool": tool_match.group(1).strip(),
        "args": args,
    }


def main() -> None:
    # sys.argv[0] 是脚本名，sys.argv[1:] 是终端传入的其余参数(在命令行终端传User Prompt)。
    # join 用空格拼接参数，strip 去掉首尾空白；空字符串为假，or 会选用默认问题。
    # 在项目目录运行示例：
    # python "1. Toolcall/02_prompt_protocol_real_model.py" "帮我查一下 Shanghai 的天气"
    user_prompt = " ".join(sys.argv[1:]).strip() or DEFAULT_USER_PROMPT

    # \n 是换行符，用来分隔终端输出的各个阶段。
    print("=== 02. Prompt 协议版 ToolCall ===")
    print("\n用户请求:")
    print(user_prompt)

    # 第 1 步：创建客户端，并通过 invoke 发出一次模型请求，等待回复。
    # 两条消息一起发给模型：System Prompt说明规则，User Prompt给出这次要解决的问题。
    # 这里传递的是工具说明文本，没有把 get_weather 函数本身交给模型执行。
    response = load_llm().invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    # response 是模型回复消息对象，content 是回复内容。
    # 本例按文本回复处理；str() 转成字符串并不等于验证模型遵守了约定格式。
    model_text = str(response.content)
    print("\n模型原始输出:")
    print(model_text)

    # 第 2 步：从模型回复中提取工具名称和参数，User Prompt中若没有涉及天气询问问题则不进行tool调用
    call = parse_tool_call(model_text)
    if call is None:
        print("\n没有工具调用，模型返回的是普通文本。")
        # return 直接结束 main，后面的工具执行代码不会运行。
        return

    print("\n解析后的工具请求:")
    print(f"tool_name = {call['tool']}")
    print(f"tool_args = {call['args']}")

    # 第 3 步：根据工具名称分发调用，只执行这里明确支持的函数。
    if call["tool"] == "get_weather":
        # call["args"] 是参数字典；** 将字典展开成关键字参数，进行关键字传参。
        # get_weather(**{"city": "Beijing"}) 等价于 get_weather(city="Beijing")。
        # 缺少 city、多出不支持的参数等情况，都可能导致调用报错。
        result = get_weather(**call["args"])
    else:
        result = f"未知工具：{call['tool']}"

    # 第 4 步：打印工具返回值。本例到此结束，不再请求模型整理最终回答。
    print("\n工具执行结果:")
    print(result)


# 直接执行本文件时运行 main；被其他模块导入时，不会自动发起模型请求。
if __name__ == "__main__":
    main()
