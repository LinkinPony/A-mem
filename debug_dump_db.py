import argparse
import chromadb
import json
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def inspect_chroma_collection(db_path: str, collection_name: str = "memories"):
    """
    Connects to a ChromaDB collection and prints all its entries for debugging.

    Args:
        db_path (str): The path to the ChromaDB database directory.
        collection_name (str): The name of the collection to inspect.
    """
    print(f"--- 正在检查 ChromaDB 数据库 ---")
    print(f"路径: {db_path}")
    print(f"集合: {collection_name}")
    print("-" * 30)

    try:
        # 1. 初始化 ChromaDB 客户端
        # 使用与您代码中相同的配置
        client_settings = Settings(allow_reset=True)
        client = chromadb.PersistentClient(path=db_path, settings=client_settings)

        # 2. 获取集合
        # 为了避免创建新集合，我们先检查它是否存在
        collections = client.list_collections()
        if not any(c.name == collection_name for c in collections):
            print(f"错误: 在 '{db_path}' 中未找到名为 '{collection_name}' 的集合。")
            print("可用的集合:", [c.name for c in collections])
            return

        embedding_function = SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        # 3. 检索所有条目
        # 使用 get() 方法获取所有内容
        all_entries = collection.get(include=["metadatas", "documents"])

        if not all_entries or not all_entries['ids']:
            print("集合为空或无法检索到条目。")
            return

        # 4. 打印每个条目
        print(f"在集合 '{collection_name}' 中找到 {len(all_entries['ids'])} 个条目。\n")

        for i, entry_id in enumerate(all_entries['ids']):
            print(f"--- 条目 {i + 1} ---")
            print(f"ID: {entry_id}")

            # 打印文档内容
            document = all_entries['documents'][i]
            print(f"文档内容: '{document}'")

            # 打印元数据，进行格式化以便阅读
            metadata = all_entries['metadatas'][i]
            print("元数据:")
            if metadata:
                for key, value in metadata.items():
                    # 元数据被保存为字符串，尝试解析 JSON
                    try:
                        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                            parsed_value = json.loads(value)
                            # 美化打印 JSON
                            formatted_value = json.dumps(parsed_value, indent=4, ensure_ascii=False)
                            print(f"  - {key}: {formatted_value}")
                        else:
                            print(f"  - {key}: {value}")
                    except (json.JSONDecodeError, TypeError):
                        print(f"  - {key}: {value} (无法解析为 JSON)")
            else:
                print("  (无元数据)")

            print("-" * 20 + "\n")

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("请确保数据库路径正确，并且您有权限访问它。")


def main():
    """
    主函数，用于解析命令行参数并调用检查函数。
    """
    parser = argparse.ArgumentParser(description="用于调试 A-Mem 项目的 ChromaDB 检查工具。")
    parser.add_argument(
        "db_path",
        type=str,
        help="ChromaDB 数据库的目录路径 (例如 './example_gemini_chroma_db')"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="memories",
        help="要检查的集合名称 (默认为 'memories')"
    )

    args = parser.parse_args()
    inspect_chroma_collection(args.db_path, args.collection)


if __name__ == "__main__":
    main()
