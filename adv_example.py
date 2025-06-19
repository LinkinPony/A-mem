import logging
import shutil
import os
import time
from memory.system import AgenticMemorySystem

logging.basicConfig(level=logging.DEBUG)

# --- Helper Function for a clean presentation ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "=" * 60)
    print(f"--- {title} ---")
    print("=" * 60)


def ask_question(memory_system, query: str, llm_controller):
    """
    Simulates an agent asking a question. It retrieves relevant memories
    and then uses an LLM to generate an answer based on that context.
    """
    print(f"\n[Q] Agent asks: {query}")

    # 1. Retrieve relevant memories
    retrieved_memories = memory_system.search_agentic(query, k=5)

    if not retrieved_memories:
        print("[A] Agent answers: I don't have enough information to answer that.")
        return

    # 2. Build context for the LLM
    context_str = "Based on the following information, please answer the user's question.\n\n"
    context_str += "--- Relevant Information ---\n"
    for mem in retrieved_memories:
        context_str += f"- Content: {mem['content']}\n"
        context_str += f"  (Context: {mem['context']})\n"
        context_str += f"  (Tags: {mem['tags']})\n\n"
    context_str += "--- End of Information ---\n\n"

    # 3. Ask the LLM to synthesize an answer
    prompt = f"{context_str}Question: {query}"

    # We use a direct call to the LLM controller for the final answer synthesis
    response = llm_controller.get_completion(prompt, stage="Question Answering")

    print(f"[A] Agent answers: {response}")


# --- 0. Initial Setup ---
print_header("Initializing Environment and Memory System")

db_path = "./complex_scenario_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)
os.makedirs(db_path, exist_ok=True)

print("Initializing AgenticMemorySystem...")
try:
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',      # Embedding model for ChromaDB
        llm_backend="gemini",               # LLM backend
        llm_model="gemini-2.5-flash",# Using a powerful and recent Gemini model
        db_path=db_path               # Use a different DB path for separate testing
    )
    print("System initialized successfully.\n")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to initialize AgenticMemorySystem: {e}")
    exit()

# --- 1. Building Character Profiles: Alex & Ben ---
print_header("Building Profiles for Alex and Ben")

# -- Alex's Profile --
print("Injecting memories about Alex...")
# Note 1: Basic Info
memory_system.add_note(
    "Alex Chen is a 32-year-old individual living in San Francisco. He has a keen interest in landscape photography and classical music."
)
time.sleep(1)
# Note 2: Career History
memory_system.add_note(
    "Regarding his career, Alex started as a software engineer at Innovate Inc. in 2015. He was promoted to Senior Software Engineer in 2018 and later moved to a Lead Engineer role at TechCorp in early 2022."
)
time.sleep(1)
# Note 3: Recent Activity
memory_system.add_note(
    "Last week, Alex mentioned he spent the entire weekend hiking at Yosemite National Park to capture the sunrise. He said the long drive was worth it."
)
time.sleep(1)

# -- Ben's Profile --
print("\nInjecting memories about Ben...")
# Note 4: Basic Info
memory_system.add_note(
    "Ben Carter, aged 35, resides in New York City. His main hobbies include marathon running and exploring modernist architecture."
)
time.sleep(1)
# Note 5: Career History
memory_system.add_note(
    "Ben Carter began his professional journey as a product manager at Solutions Co. back in 2016. He stayed there for five years before co-founding his own startup, 'ConnectSphere', in 2021 where he acts as CEO."
)
time.sleep(1)
# Note 6: Recent Activity
memory_system.add_note(
    "Over this past weekend, Ben flew to Chicago for a conference. He also managed to squeeze in a visit to the Art Institute of Chicago and tried their famous deep-dish pizza."
)
time.sleep(1)

print("\n--- All memories have been added and processed. ---")

# --- 2. Answering Questions based on the Evolved Knowledge ---
print_header("Querying the Agentic Memory")

# A dedicated LLM controller for the "Answering" agent
qa_controller = memory_system.llm_controller

# Question 1: Simple Retrieval
ask_question(memory_system, "What company did Alex Chen work for before TechCorp?", qa_controller)

# Question 2: Information Synthesis
ask_question(memory_system, "Who is older, Alex or Ben, and what are their respective hobbies?", qa_controller)

# Question 3: Temporal Reasoning & Exclusion
ask_question(memory_system, "Did anyone go hiking in New York last week?", qa_controller)

# Question 4: Complex Inference
ask_question(memory_system,
             "Which person is more likely to be interested in a new productivity app for startup founders, and why?",
             qa_controller)

print("\n--- Example script finished. ---")
