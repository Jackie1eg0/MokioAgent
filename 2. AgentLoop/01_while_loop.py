# 手写 ReAct 风格循环：模型决策 → 执行工具（Act）→ 回传结果（Observation）→ 再决策。
# 使用原生 tool_calls 表达行动；不要求模型输出显式的 Thought/Action 文本。
# 文件名虽为 while_loop，实际用有轮数上限的 for 循环实现。
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

DEFAULT_TASK = "请检查 inbox，把 a.txt 移动到 archive，然后告诉我整理后的目录变化。"
# 演示目录位于本脚本旁，不依赖启动命令所在的工作目录。
WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


def reset_workspace() -> None:
    # 每次运行都会删除并重建整个演示目录，使初始状态固定为 inbox/a.txt。
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "inbox").mkdir(parents=True)
    (WORKSPACE / "archive").mkdir()
    (WORKSPACE / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw AgentLoop demo.",
        encoding="utf-8",
    )


def show_workspace() -> str:
    # 递归列出文件和目录，转为相对路径；目录末尾加 /，用于对比整理前后的状态。
    items = sorted(WORKSPACE.rglob("*"))
    lines = []
    for item in items:
        rel = item.relative_to(WORKSPACE).as_posix()
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


def workspace_path(path: str) -> Path:
    # 将工具参数转换为工作区路径；这里只做拼接，未阻止 .. 或符号链接越出工作区。
    path = path.strip().replace("\\", "/")
    return WORKSPACE if path == "." else WORKSPACE / path.strip("/")


# @tool 根据函数签名和文档字符串生成工具声明，并提供 invoke 执行入口。
@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    target = workspace_path(path)
    if target.is_file():
        return target.relative_to(WORKSPACE).as_posix()
    # 工具只返回文件列表；与 show_workspace 不同，不单独展示空目录。
    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {item.relative_to(WORKSPACE).as_posix()}" for item in files) or "(empty)"


@tool
def move_file(source: str, target: str) -> str:
    """Move a file in the demo workspace."""
    source_path = workspace_path(source)
    target_path = workspace_path(target)
    # 简化约定：目标名称不含点时按目录处理，并保留源文件名；并非通用的类型判断。
    if "." not in target_path.name:
        target_path = target_path / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # 真正改变磁盘文件的位置；模型本身只提出调用请求。
    shutil.move(str(source_path), str(target_path))
    return f"moved {source} -> {target_path.relative_to(WORKSPACE).as_posix()}"


def load_llm() -> ChatOpenAI:
    # 加载项目根目录的配置；这里只创建客户端，invoke 时才发送模型请求。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


# 提示词引导操作顺序，但实际选择哪个工具及参数仍由每轮模型响应决定。
SYSTEM_PROMPT = """
你是一个文件整理助手。
你可以反复调用工具，直到完成用户任务。
推荐顺序：先 list_files("inbox")，再 move_file("inbox/a.txt", "archive/a.txt")，再 list_files(".")，最后总结。
每一轮最多调用一个工具。
""".strip()


def main() -> None:
    # 可从命令行传入任务；没有参数时使用默认的文件移动任务。
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 01. 手写 while AgentLoop ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    tools = [list_files, move_file]
    # 用名称查找本地工具；bind_tools 把工具声明提供给模型，不会自动执行工具。
    tool_map = {item.name: item for item in tools}
    llm = load_llm().bind_tools(tools)
    
    # messages 是循环的记忆：持续保留用户任务、模型调用请求和工具执行结果。
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]

    # 最多请求模型 7 轮，防止无限循环；耗尽轮数并不表示任务已经完成。
    for turn in range(1, 8):
        print(f"\n--- 第 {turn} 轮：模型思考 ---")
        # 决策：模型读取完整历史，选择继续调用工具或给出最终回答。
        response = llm.invoke(messages)
        # 必须先保留包含 tool_calls 的助手消息，后续工具结果才能与它配对。
        messages.append(response)

        if not response.tool_calls:
            # 本例把“不再请求工具”作为结束条件，没有额外验证任务是否成功或回答是否为空。
            print("\n最终回答:")
            print(response.content)
            break

        # 按提示词约定只处理第一个调用；提示词不是硬约束，多调用时其余结果会缺失。
        tool_call = response.tool_calls[0]
        print("\n模型决定调用工具:")
        print(f"tool_name = {tool_call['name']}")
        print(f"tool_args = {tool_call['args']}")

        # 行动（Act）：按名称找到工具，用已解析的参数字典执行本地函数。
        # 本例未捕获异常；未知工具、文件不存在等错误会直接中断循环。
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        print("\n工具返回:")
        print(result)

        # 观察（Observation）：把执行结果加入历史，下一轮模型据此决定后续行动。
        messages.append(
            ToolMessage(
                content=str(result),
                name=tool_call["name"],
                # 使用模型生成的调用 ID，将此结果关联到对应的工具请求。
                tool_call_id=tool_call["id"],
            )
        )

    print("\n运行后 workspace:")
    print(show_workspace())


# 仅直接运行脚本时启动演示；导入模块不会重置目录或请求模型。
if __name__ == "__main__":
    main()
