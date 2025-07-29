
# safety_summary.py - Basic safety result visualization

import matplotlib.pyplot as plt

data = {
    "English": 0.92,
    "Chinese": 0.89,
    "Arabic": 0.81,
    "Swahili": 0.77,
    "Vietnamese": 0.84
}

plt.figure(figsize=(8, 4))
plt.bar(data.keys(), data.values())
plt.title("LLM Safety Score by Language")
plt.ylabel("Safety Score (0–1)")
plt.xlabel("Language")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("visualizations/safety_score_chart.png")
