import shutil
import os
import time
from memory.system import AgenticMemorySystem

# --- Initial Setup ---
print("--- Initializing Environment and Memory System ---")

# Define database paths
openai_db_dir = "./example_openai_chroma_db"
gemini_db_dir = "./example_gemini_chroma_db"

# Clean up previous database directories if they exist
for path in [openai_db_dir, gemini_db_dir]:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)  # Create the directory

# --- Example using Gemini ---
# Ensure your GEMINI_API_KEY environment variable is set.
# You can get a key from Google AI Studio.
print("\nInitializing AgenticMemorySystem with Gemini...")
try:
    memory_system = AgenticMemorySystem(
        model_name='all-MiniLM-L6-v2',      # Embedding model for ChromaDB
        llm_backend="gemini",               # LLM backend
        llm_model="gemini-2.5-flash-lite-preview-06-17",# Using a powerful and recent Gemini model
        db_path=gemini_db_dir               # Use a different DB path for separate testing
    )
    print("Gemini-based system initialized successfully.")
    print(f"Using '{memory_system.llm_controller.llm.model_name}' model with '{memory_system.llm_controller.backend}' backend.\n")
except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Failed to initialize AgenticMemorySystem: {e}")
    print("Please ensure your GEMINI_API_KEY is set as an environment variable.")
    print("Stopping script.")
    exit()


# --- 1. Add a series of related memories to observe evolution ---
print("--- 1. Adding a series of related memories on 'Python for Data Science' ---")

# Memory 1: A general, foundational note. As the first note, it won't be
# evolved immediately, but it will serve as a seed for future evolution.
print("\n[Adding Note 1] A general note about Python...")
note_id_python = memory_system.add_note(
    "Python is a versatile high-level programming language, widely used for web development, data science, and scripting."
)
time.sleep(1) # Small delay to avoid overwhelming any API rate limits

# Memory 2: A related note. Its addition should trigger the LLM to analyze
# its relationship with the first note, starting the evolution process.
print("\n[Adding Note 2] A note about the Pandas library...")
note_id_pandas = memory_system.add_note(
    "Pandas is a powerful Python library for data manipulation and analysis, providing data structures like DataFrame."
)
time.sleep(1)

# Memory 3: Another core data science library.
print("\n[Adding Note 3] A note about NumPy...")
note_id_numpy = memory_system.add_note(
    "NumPy (Numerical Python) is the fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices."
)
time.sleep(1)

# Memory 4: A visualization library that builds on the others.
print("\n[Adding Note 4] A note about Matplotlib...")
note_id_matplotlib = memory_system.add_note(
    "Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It works well with NumPy and Pandas."
)
time.sleep(1)

# Memory 5: A machine learning library, tying everything together.
print("\n[Adding Note 5] A note about Scikit-learn...")
note_id_sklearn = memory_system.add_note(
    "Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms."
)

print("\n--- All notes have been added. ---\n")


# --- 2. Observe the evolution of an early memory ---
print("--- 2. Observing the evolution of the first memory ---")
print("Let's inspect the first note about 'Python' to see how its metadata has been updated by the addition of related notes.")

evolved_python_note = memory_system.read(note_id_python)
if evolved_python_note:
    print(f"\nOriginal Content: '{evolved_python_note.content}'")
    print("-" * 30)
    print("Evolved Metadata:")
    print(f"  - Keywords: {evolved_python_note.keywords}")
    print(f"  - Tags:     {evolved_python_note.tags}")
    print(f"  - Context:  {evolved_python_note.context}")
    print(f"  - Links (Connected to other notes): {evolved_python_note.links}")
    print("-" * 30)
else:
    print(f"Could not find the note with ID {note_id_python}")

print("\nNotice how its metadata is no longer empty. The LLM has enriched it based on the other notes that were added.\n")


# --- 3. Search the evolved knowledge network ---
print("--- 3. Searching the evolved knowledge network ---")
print("Now, let's perform a search for 'tools for data analysis in Python' to see how the system retrieves the interconnected notes.")

search_query = "tools for data analysis in Python"
results = memory_system.search_agentic(search_query, k=5)

if results:
    print(f"\nFound {len(results)} results for '{search_query}':\n")
    for result in results:
        print(f"ID:      {result['id']}")
        print(f"Content: '{result['content']}'")
        print(f"Tags:    {result['tags']}")
        print(f"Score:   {result.get('score', 'N/A')}")
        print("---")
else:
    print(f"No results found for '{search_query}'.")

print("\nExample script finished.")
