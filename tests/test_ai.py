import pandas as pd

from src.ai_agent import ask_ai


# ============================================================
# LOAD SAMPLE DATA
# ============================================================

df = pd.read_csv(
    "data/sample.csv"
)


# ============================================================
# ASK DATAPILOT AI
# ============================================================

question = input(
    "\n🤖 Ask DataPilot AI something about your dataset: "
)


answer = ask_ai(
    question,
    df
)


# ============================================================
# DISPLAY ANSWER
# ============================================================

print("\n🤖 DataPilot AI:")
print(answer)