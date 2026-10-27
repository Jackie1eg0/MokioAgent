# 本文件用于观察：直接使用 OpenAI Python SDK 时，模型的原生 tool call 长什么样。
# 整体流程：加载配置 → 声明工具接口 → 向模型提问 → 打印完整响应 → 提取工具名和参数。
# 模型返回的 tool call 只是“希望调用哪个工具、传什么参数”的请求，不是工具执行结果。
# 与 03_langchain_native_toolcall.py 相比：这里直接查看 SDK 响应，不经过 LangChain；

from __future__ import annotations


import json
import os
from pathlib import Path
from dotenv import load_dotenv
# 模型接口客户端；也可通过 base_url 连接兼容 OpenAI 接口格式的其他服务。
from openai import OpenAI


def main() -> None:
    # load_dotenv 将配置载入环境变量；默认不会覆盖进程中已经存在的同名变量。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model=os.getenv("MODEL")
    # 创建同步客户端：这里只配置连接，真正的模型请求发生在下面的 create()。
    # os.getenv("名称") 读取环境变量，变量不存在时返回 None。
    client = OpenAI(
        # 指定模型服务的 API 基础地址。
        base_url=os.getenv("BASE_URL"),
        # 指定模型服务的认证密钥。
        api_key=os.getenv("API_KEY"),
    )

    # 发起一次 Chat Completions 请求，并等待完整响应（本例没有启用流式输出）。
    # 发送的是“用户问题 + 工具说明及参数规则”，不是 Python 函数代码。
    response = client.chat.completions.create(
        model=model,
        # messages 是对话消息列表；此处只有一条用户消息，没有系统消息或历史记录。
        messages=[{"role": "user", "content": "北京天气怎么样？"}],
        # tools 是提供给模型的可用工具声明列表。本例只声明一个函数工具。
        tools=[
            {
                # function 表示由调用方提供实现的函数工具。
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询指定城市的天气。",
                    # 请求服务端严格遵守下面的参数结构；是否支持取决于服务实现,strict 约束参数格式
                    "strict": True,
                    "parameters": {
                        # 工具参数整体应为 JSON 对象，例如 {"city": "Beijing"}。
                        "type": "object",
                        # properties 列出允许的参数名称及各自的类型、说明。
                        "properties": {
                            "city": {
                                "type": "string",
                                # 这是自然语言提示，帮助模型把“北京”转换成 Beijing。
                                "description": "城市英文名，例如 Beijing、Shanghai。",
                            }
                        },
                        # city 是必填参数。
                        "required": ["city"],
                        # 不允许生成 properties 中未声明的额外参数。
                        "additionalProperties": False,
                    },
                },
            }
        ],
        # 较低温度通常让输出更稳定；设为 0 也不保证每次输出完全相同。
        temperature=0,
    )

    print("=== API 原始返回值 ===")
    # response 是 SDK 解析后的响应对象，通常包含 id、model、choices、usage 等字段。
    # model_dump_json 将整个对象重新序列化为 JSON 文本；indent=2 使每层缩进两个空格。
    print(response.model_dump_json(indent=2))

    print("\n=== 单独取出模型生成的 tool call ===")
    # choices 是候选回复列表；[0] 取第一条，本例假定服务返回了至少一个候选回复。
    # message 是该候选中的助手消息，可能包含文本 content，也可能包含 tool_calls。
    message = response.choices[0].message
    # 如果 tool_calls 为 None 或空列表，统一使用 []，方便后续遍历和判断。
    tool_calls = message.tool_calls or []

    # 一条助手消息可能包含多个工具调用请求，因此逐个遍历，而不是只取第一个。
    for tool_call in tool_calls:
        # id 标识这一次调用；
        print(f"id        = {tool_call.id}")
        # type 在本例中应为 "function"。
        print(f"type      = {tool_call.type}")
        # name 是模型选择的工具名，例如 "get_weather"。
        print(f"name      = {tool_call.function.name}")
        # arguments 是模型生成的 JSON 字符串，例如 '{"city":"Beijing"}'，尚非字典。
        # 与 LangChain 整理后的 tool_call["args"] 不同，这里需要自己解析 JSON。
        print(f"arguments = {tool_call.function.arguments}")

        print("\n=== arguments 解析成 Python dict 后 ===")
        # 下面从内到外执行两次转换：
        # 1. json.loads：JSON 字符串 → Python 对象；符合本例 object 结构时得到 dict。
        # 2. json.dumps：Python 对象 → 排版后的 JSON 字符串，方便终端阅读。
        print(
            json.dumps(
                json.loads(tool_call.function.arguments),
                ensure_ascii=False,
                indent=2,
            )
        )

    if not tool_calls:
        # 没有工具调用时输出提示。模型也可能只是选择直接回答，并不一定是不支持工具。
        # 若存在普通文本回复，可在上面完整响应中的 choices[0].message.content 查看。
        print("没有拿到 tool_calls。请确认当前模型支持 Chat Completions tool calling。")


# 直接运行脚本时执行 main；被其他模块 import 时，不会自动发送模型请求。
if __name__ == "__main__":
    main()
