import shutil
import os
import time
from memory.system import AgenticMemorySystem


def print_header(title: str):
    """打印一个格式清晰的标题。"""
    print("\n" + "=" * 70)
    print(f"--- {title} ---")
    print("=" * 70)


def ask_question(memory_system: AgenticMemorySystem, query: str):
    """
    模拟一个 Agent 提出问题，检索相关记忆，并综合信息进行回答。
    """
    print(f"\n[Q] Agent 提问: {query}")

    # 1. 从 A-Mem 检索与问题最相关的记忆片段
    retrieved_memories = memory_system.search_agentic(query, k=10)  # 检索更多信息以获得更全的上下文

    if not retrieved_memories:
        print("[A] Agent 回答: 我的记忆中没有足够的信息来回答这个问题。")
        return

    # 2. 构建用于最终回答的上下文
    context_str = "请基于以下背景信息，用中文详细回答问题。\n\n--- 背景信息 ---\n"
    for i, mem in enumerate(retrieved_memories):
        context_str += f"记忆片段 {i + 1}:\n"
        context_str += f"- 内容: {mem.get('content', 'N/A')}\n"
        context_str += f"- 标签: {mem.get('tags', [])}\n\n"
    context_str += "--- 信息结束 ---\n\n"

    # 3. 使用一个独立的 LLM 调用来合成最终答案
    # 这模拟了 Agent 使用记忆系统作为其“外脑”的真实工作流程
    prompt = f"{context_str}问题: {query}"

    response = memory_system.llm_controller.get_completion(prompt, stage="Final Answer Synthesis")

    print(f"[A] Agent 回答:\n{response}")


# --- 0. 环境初始化 ---
print_header("初始化环境与 A-Mem 系统")

db_path = "./multi_character_scenario_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
os.makedirs(db_path, exist_ok=True)

print("正在初始化 AgenticMemorySystem...")
try:
    memory_system = AgenticMemorySystem(
        llm_backend="gemini",
        llm_model="gemini-2.5-flash-lite-preview-06-17",  # 推荐使用能力更强的模型以获得更好的推理结果
        db_path=db_path
    )
    print("系统初始化成功。")
except Exception as e:
    print(f"\n--- 错误 ---")
    print(f"初始化 AgenticMemorySystem 失败: {e}")
    exit()

# --- 1. 构建四个人物画像 ---
print_header("第一阶段: 构建四个人物画像的记忆网络")

# -- 人物 1: Dr. Aris Thorne (考古学家) --
print("\n注入关于考古学家 Aris 的记忆...")
memory_system.add_note("Aris Thorne 是一位严谨的考古学家，专注于研究古埃及象形文字。")
memory_system.add_note("Aris 对咖啡非常挑剔，只喝单一来源的埃塞俄比亚耶加雪菲。")
memory_system.add_note("上个月，Aris 刚从一次为期三周的埃及帝王谷实地考察中回来。")
memory_system.add_note("Aris 认为，准确的数据记录是考古工作的基石，他鄙视任何形式的猜测。")
# 长描述
memory_system.add_note(
    "Aris Thorne 在整理从埃及带回的陶器碎片时遇到了一个难题。他发现其中一些碎片上的标记既不完全符合已知的第十八王朝的任何一种象形文字变体，也与邻近时期的风格有明显出入。他怀疑这可能是一种非常罕见的、地方性的速记文字，或者是某个特定工匠群体的独特标记。为了验证这个假设，他正在 painstakingly 地将数千个碎片的拓片数字化，并计划使用模式识别软件进行交叉比对，这项工作非常耗时。"
)

# -- 人物 2: Maya Singh (金融分析师) --
print("\n注入关于金融分析师 Maya 的记忆...")
memory_system.add_note("Maya Singh 是一名在伦敦工作的快节奏金融分析师，专门负责科技股板块。")
memory_system.add_note("Maya 的办公桌上总是放着一个专业的彭博终端机。")
memory_system.add_note("Maya 上周参加了一个关于可持续投资的全球金融峰会。")
memory_system.add_note("尽管工作繁忙，Maya 坚持每天早上进行一小时的瑜伽来保持专注。")
# 长描述
memory_system.add_note(
    "Maya Singh 目前正在为一个大客户准备一份关于半导体行业的深度分析报告。她认为，尽管当前市场因为供应链问题而有所波动，但长期来看，专注于AI芯片和汽车半导体的公司拥有巨大的增长潜力。她的报告核心论点是，投资者应该超越短期的市场噪音，关注那些拥有强大研发能力和专利护城河的公司。她正在使用复杂的金融模型来预测未来五年的现金流，并对几个主要目标公司进行估值。"
)

# -- 人物 3: Leo Vance (平面设计师) --
print("\n注入关于平面设计师 Leo 的记忆...")
memory_system.add_note("Leo Vance 是一位居住在柏林的自由平面设计师，以其极简主义和大胆的色彩运用而闻名。")
memory_system.add_note("Leo 的主要创作工具是 iPad Pro 上的 Procreate 应用和一台高配的 MacBook Pro。")
memory_system.add_note("Leo 最近完成了一个为 Dr. Kenji Tanaka 的生物研究实验室设计的 Logo。")
memory_system.add_note("Leo 透露他正在学习三维建模，希望将 3D 元素融入到他未来的作品中。")
# 长描述
memory_system.add_note(
    "Leo Vance 对当前设计界过度依赖AI生成工具的趋势感到担忧。他认为，虽然AI可以快速产出大量看似不错的图像，但它们往往缺乏真正的创意火花和深层次的文化内涵。在他看来，一个好的设计是设计师个人经验、情感和对客户需求的深刻理解的结晶。他正在撰写一篇博客文章，主张设计师应该将AI视为一个辅助工具，而不是创意的来源，并强调手绘草图和概念构思在设计流程中不可替代的重要性。"
)

# -- 人物 4: Dr. Kenji Tanaka (生物学家) --
print("\n注入关于生物学家 Kenji 的记忆...")
memory_system.add_note("Dr. Kenji Tanaka 是一位专注于海洋微生物基因组学研究的生物学家。")
memory_system.add_note("Kenji 的实验室最近因为在深海热泉附近发现了一种全新的耐高温细菌而获得了业界关注。")
memory_system.add_note("Kenji 提到，他的团队在进行基因测序时，主要依赖于高性能计算集群来处理海量数据。")
memory_system.add_note("Kenji 平日里喜欢通过看科幻电影来放松，尤其是那些关于太空探索的。")
# 长描述
memory_system.add_note(
    "Dr. Kenji Tanaka 的团队目前正面临一个巨大的挑战：他们新发现的耐高温细菌的基因组中，有大约30%的基因序列与已知数据库中的任何生物都无法匹配。这部分“暗物质基因”的功能完全未知，Kenji 推测它们可能编码了使这种细菌能够在极端环境下生存的独特蛋白质和代谢通路。为了解开这个谜团，他们正在设计复杂的蛋白质折叠模拟实验，并计划与一个拥有更强大计算资源的超算中心合作。"
)

print("\n--- 所有记忆已注入并完成初步演化 ---")

# --- 2. 提出复杂问题以测试系统能力 ---
print_header("第二阶段: 对记忆网络进行复杂查询与推理")

# 问题 1: 多跳信息综合与比较
ask_question(memory_system, "比较一下 Leo 和 Dr. Tanaka 在工作中最依赖的专业工具分别是什么？")

# 问题 2: 动机与挑战的深层推理
ask_question(memory_system, "Dr. Thorne 最近的埃及之行目的是什么？他当前在研究中面临的具体挑战是什么？")

# 问题 3: 带有干扰信息的逻辑排除
ask_question(memory_system, "上个月是否有人去柏林参加了科技行业的会议？")

# 问题 4: 跨人物关系推断
ask_question(memory_system, "Leo Vance 和 Dr. Kenji Tanaka 之间有什么样的专业联系？")

# 问题 5: 基于性格画像的预测性问题
ask_question(memory_system, "在这四个人中，谁最有可能订阅一份关于'古代文明未解之谜'的付费考古学杂志？请解释原因。")

print("\n--- 复杂场景测试脚本执行完毕 ---")

