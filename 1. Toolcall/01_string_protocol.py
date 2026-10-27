# 本文件演示最简单的 Tool Call（工具调用）：用一段约定格式的字符串表达调用请求。
# 可以把“工具”理解成一个普通 Python 函数，例如下面的 get_weather。
# 核心流程：模型生成工具的请求文本 → 程序解析文本 → 程序调用函数 → 得到工具结果。
# 注意：模型只负责“提出调用请求”，真正执行 Python 函数的是我们的程序。

# 这里用固定字符串模拟模型输出，没有连接真实大模型，也没有查询实时天气，
# 更没有把工具结果交回模型生成最终回答；本例只演示解析和执行两个环节。

from __future__ import annotations


# city: str 表示期望接收参数 city 是字符串；-> str 表示函数返回字符串。
def get_weather(city: str) -> str:
    # 用字典模拟天气数据：键是城市名称，值是对应的天气描述。
    # 城市名称必须与这里的键匹配，例如 "Beijing"，而不是 "北京"。
    weather = {
        "Beijing": "晴天，25 度",
        "Shanghai": "多云，28 度",
        "Hangzhou": "小雨，22 度",
    }
    # dict.get(键, 默认值)：找到城市就返回天气，找不到就返回“未知天气”。
    # f"...{表达式}..." 是 f-string，会把花括号中的计算结果填入字符串。
    return f"{city} 的天气是：{weather.get(city, '未知天气')}"


# 解析模型请求工具调用的输出，LLM需要说明调用的tool名称、tool传入的参数。此处假设为"get_weather:location"
# 返回值解析LLM的工具请求 -> (tool_name,tool_args) 其中tool_args是dict类型，键值均为str类型。
def parse_model_output(text: str) -> tuple[str, dict[str, str]]:
    # split 按冒号拆分；maxsplit=1 表示最多拆一次，得到工具名和城市名两部分。
    # 例如输入 "get_weather:Beijing"，返回 ("get_weather", {"city": "Beijing"})。
    # 如果拆分后不足两项，解包就会抛出 ValueError。
    tool_name, city = text.split(":", maxsplit=1)
    # strip() 去掉首尾空白；字典中的 "city" 必须与 get_weather 的参数名一致。
    return tool_name.strip(), {"city": city.strip()}


# main 把整个演示串起来；-> None 表示这个函数不返回有意义的结果。
def main() -> None:
    print("=== 01. 字符串协议版 ToolCall ===")

    # 第 1 步：模拟LLM请求工具调用的输出。实际接入模型时，需要事先告诉模型按这个协议输出。
    model_output = "get_weather:Beijing"

    # 字符串中的 \n 表示换行，让终端输出的各个步骤更容易区分。
    print("\n假设LLM模型输出:")
    print(model_output)

    # 第 2 步：把文本转换成程序能使用的工具名称和参数字典。
    # “get_weather:Beijing" -> ("getweather", {"city":"Beijing"})
    tool_name, tool_args = parse_model_output(model_output)
    
    print("\n解析后的工具请求:")
    print(f"tool_name = {tool_name}")
    print(f"tool_args = {tool_args}")

    # 第 3 步：根据工具名称，选择并执行对应函数（也称“工具分发”）。
    # 这里演示明确只允许调用 get_weather；模型输出其他名称时，进入 else 分支。
    if tool_name == "get_weather":
        # ** 把字典展开成关键字参数：相当于关键字传参,字典的键作为关键字，值作为该关键字对应参数。
        # get_weather(**{"city": "Beijing"}) 等价于 get_weather(city="Beijing")。
        # 字典的键必须匹配函数参数名，否则调用可能抛出 TypeError。
        result = get_weather(**tool_args)
    else:
        result = f"未知工具：{tool_name}"

    # 第 4 步：展示工具返回的结果，本例到这里结束。
    # 在常见的大模型工具调用流程中，还会把结果传回模型，让模型据此回答用户。
    print("\n工具执行结果:")
    print(result)


# 直接运行本文件时，__name__ 为 "__main__"，于是执行 main()。
# 如果本文件被其他模块导入，则不会自动运行演示，仍然可以使用上面定义的函数。
if __name__ == "__main__":
    main()
