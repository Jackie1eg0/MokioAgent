from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 默认User Prompt输入
DEFAULT_TASK = "请检查 inbox，把 a.txt 移动到 archive，然后告诉我整理后的目录变化。"
WORKSPACE = Path(__file__).resolve().parent / "demo_workspace"


def reset_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    (WORKSPACE / "inbox").mkdir(parents=True)
    (WORKSPACE / "archive").mkdir()
    (WORKSPACE / "inbox" / "a.txt").write_text(
        "Hello from MokioClaw AgentLoop demo.",
        encoding="utf-8",
    )


def show_workspace() -> str:
    items = sorted(WORKSPACE.rglob("*"))
    lines = []
    for item in items:
        rel = item.relative_to(WORKSPACE).as_posix()
        lines.append(f"- {rel}/" if item.is_dir() else f"- {rel}")
    return "\n".join(lines) or "(empty)"


def workspace_path(path: str) -> Path:
    path = path.strip().replace("\\", "/")
    return WORKSPACE if path == "." else WORKSPACE / path.strip("/")

# LangChain中的@tool装饰器封装函数作为BaseTool对象，用于tool call
# """..."""中间为tool description
@tool
def list_files(path: str = ".") -> str:
    """List files in the demo workspace."""
    target = workspace_path(path)
    if target.is_file():
        return target.relative_to(WORKSPACE).as_posix()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    return "\n".join(f"- {item.relative_to(WORKSPACE).as_posix()}" for item in files) or "(empty)"


@tool
def move_file(source: str, target: str) -> str:
    """Move a file in the demo workspace."""
    source_path = workspace_path(source)
    target_path = workspace_path(target)
    if "." not in target_path.name:
        target_path = target_path / source_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(target_path))
    return f"moved {source} -> {target_path.relative_to(WORKSPACE).as_posix()}"


def load_llm() -> ChatOpenAI:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return ChatOpenAI(
        model=os.getenv("MODEL", "qwen3.6-flash"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0,
    )


SYSTEM_PROMPT = """
你是一个 ReAct 文件整理助手。
ReAct 的节奏是：观察当前状态 -> 选择一个工具行动 -> 查看工具结果 -> 决定下一步。
目标：检查 inbox，移动 inbox/a.txt 到 archive/a.txt，查看整理后的目录，然后总结。
每一轮最多调用一个工具。
""".strip()


def main() -> None:
    from langchain.agents import create_agent
    
    # 从命令行参数中获取User Prompt，没有使用DEFAULT_TASK
    task = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    reset_workspace()

    print("=== 02. LangChain create_agent：封装好的 ReAct Loop ===")
    print("\n用户任务:")
    print(task)
    print("\n运行前 workspace:")
    print(show_workspace())

    agent = create_agent(
        model=load_llm(),
        tools=[list_files, move_file],
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})

    print("\nAgent 消息流:")
    for message in result["messages"]:
        print(f"\n[{getattr(message, 'type', 'unknown')}]")
        if getattr(message, "tool_calls", None):
            print(message.tool_calls)
        elif getattr(message, "content", ""):
            print(message.content)

    print("\n运行后 workspace:")
    print(show_workspace())


if __name__ == "__main__":
    main()
