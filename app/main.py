"""多Agent智能体系统 - CLI入口"""
import uuid
import os
import sys

# readline 解决终端输入清除/光标异常问题
try:
    import readline  # noqa: F401
except ImportError:
    pass

from app.agents.router import RouterAgent
from app.rag.vector_store import ChromaVectorStore
from app.memory.memory import PersistentMemory
from app.memory.shared_memory import SharedMemory


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("       多Agent智能体系统 v1.0")
    print("=" * 60)
    print("\n可用命令：")
    print("  /upload <文件路径>    上传文档到知识库")
    print("  /agents              查看可用Agent列表")
    print("  /history             查看对话历史")
    print("  /stats               查看系统统计")
    print("  /clear               清除当前会话")
    print("  /quit                退出系统")
    print("\n直接输入问题即可开始对话，系统会自动路由到合适的Agent。")
    print("-" * 60 + "\n")


def handle_upload(args: str, vector_store: ChromaVectorStore):
    """处理文档上传"""
    file_path = args.strip()
    if not file_path:
        print("[系统] 请指定文件路径，例如: /upload ./data/documents/公司手册.txt")
        return

    # 支持相对路径
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"[系统] 文件不存在: {file_path}")
        return

    print(f"[系统] 正在处理文档: {os.path.basename(file_path)}...")
    try:
        num_chunks = vector_store.add_file(file_path)
        print(f"[系统] 文档上传成功！已分割为 {num_chunks} 个文本块并存入知识库。")
    except Exception as e:
        print(f"[系统] 上传失败: {str(e)}")


def handle_agents(router: RouterAgent):
    """显示可用Agent列表"""
    print("\n可用Agent列表：")
    print("-" * 40)
    agents = router.get_available_agents()
    for key, desc in agents.items():
        print(f"  • {desc} ({key})")
    print("-" * 40)
    print()


def handle_history(memory: PersistentMemory, session_id: str):
    """显示对话历史"""
    history = memory.get_full_session_history(session_id)
    if not history:
        print("[系统] 当前会话暂无对话历史。")
        return

    print(f"\n对话历史（共 {len(history)} 条）：")
    print("-" * 40)
    for msg in history[-20:]:  # 最多显示最近20条
        agent = msg.get('agent_name', 'unknown')
        role = "👤 用户" if msg['role'] == 'user' else f"🤖 {agent}"
        content = msg['content'][:200]
        timestamp = msg.get('timestamp', '')
        print(f"  [{timestamp}] {role}: {content}")
        if len(msg['content']) > 200:
            print(f"    ... (截断)")
    print("-" * 40)
    print()


def handle_stats(vector_store: ChromaVectorStore, memory: PersistentMemory, session_id: str):
    """显示统计信息"""
    print("\n系统统计：")
    print("-" * 40)

    # 知识库统计
    stats = vector_store.get_collection_stats()
    print(f"  知识库文档块数: {stats['count']}")
    print(f"  存储目录: {stats['persist_dir']}")

    # 文档来源
    sources = vector_store.list_sources()
    if sources:
        print(f"  已导入文档: {', '.join(sources)}")

    # 会话统计
    session_stats = memory.get_session_stats(session_id)
    print(f"  当前会话消息数: {session_stats.get('total', 0)}")
    print(f"  使用的Agent数: {session_stats.get('agents_used', 0)}")

    print("-" * 40)
    print()


def handle_clear(memory: PersistentMemory, shared_memory: SharedMemory, session_id: str):
    """清除当前会话"""
    memory.clear_session(session_id)
    shared_memory.clear_session(session_id)
    print("[系统] 当前会话已清除。")


def main():
    """主入口"""
    # 生成会话ID
    session_id = str(uuid.uuid4())

    # 初始化组件
    try:
        router = RouterAgent()
        vector_store = ChromaVectorStore()
        memory = PersistentMemory()
        shared_memory = SharedMemory()
    except Exception as e:
        print(f"[错误] 系统初始化失败: {str(e)}")
        print("请检查配置文件 .env 中的设置是否正确。")
        sys.exit(1)

    # 显示欢迎
    print_welcome()
    print(f"[系统] 会话已启动 (ID: {session_id[:8]}...)")
    print()

    # 主循环
    while True:
        try:
            user_input = input("你: ")
            # 清理不可见字符和首尾空白
            user_input = user_input.strip().replace("\x00", "").replace("\u200b", "")

            if not user_input:
                continue

            # 命令处理
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command in ["/quit", "/exit", "/q"]:
                    print("\n[系统] 再见！")
                    break
                elif command == "/upload":
                    handle_upload(args, vector_store)
                elif command == "/agents":
                    handle_agents(router)
                elif command == "/history":
                    handle_history(memory, session_id)
                elif command == "/stats":
                    handle_stats(vector_store, memory, session_id)
                elif command == "/clear":
                    handle_clear(memory, shared_memory, session_id)
                else:
                    print(f"[系统] 未知命令: {command}")
                    print("  输入 /help 或查看上方命令列表")
                continue

            # 普通对话 - 路由到Agent
            print()  # 空行分隔
            result = router.route(user_input, session_id)
            agent_name = result.get("agent_name", "未知")
            response = result.get("response", "无响应")

            print(f"[{agent_name}]: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\n[系统] 再见！")
            break
        except EOFError:
            print("\n[系统] 再见！")
            break
        except Exception as e:
            print(f"\n[错误] {str(e)}")
            print("[系统] 发生异常，但系统仍可继续使用。\n")


if __name__ == "__main__":
    main()
