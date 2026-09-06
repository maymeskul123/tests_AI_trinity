import csv
from pathlib import Path
from collections import defaultdict

# Путь к исходному CSV
input_csv = Path("results/phase3/test08/results.csv")
output_csv = Path("results/phase3/test08/summary.csv")

if not input_csv.exists():
    print(f"Error: {input_csv} not found. Run test08 first.")
    exit(1)

# Читаем данные
rows = []
with input_csv.open("r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# Группировка по ключевым параметрам
groups = defaultdict(list)
for row in rows:
    key = (
        row["policy"],
        float(row["confidence_threshold"]),
        int(row["n_candidates"]),
        int(row["n_obs_initial"]),
        float(row["p_true_support"]),
        float(row["p_false_support"]),
    )
    groups[key].append(row)

# Агрегация
summary_rows = []
for (policy, threshold, n_cand, n_obs, p_true_s, p_false_s), group in groups.items():
    total = len(group)
    decisions = sum(1 for r in group if r["decision"] != "ABSTAIN")
    correct = sum(int(r["correct"]) for r in group)
    total_queries = sum(int(r["queries"]) for r in group)
    accuracy = correct / decisions if decisions > 0 else 0.0
    coverage = decisions / total
    avg_queries = total_queries / total

    summary_rows.append({
        "policy": policy,
        "confidence_threshold": threshold,
        "n_candidates": n_cand,
        "n_obs_initial": n_obs,
        "p_true_support": p_true_s,
        "p_false_support": p_false_s,
        "total_scenarios": total,
        "decisions": decisions,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "coverage": round(coverage, 4),
        "avg_queries": round(avg_queries, 3),
    })

# Сортируем для удобства
summary_rows.sort(key=lambda x: (x["policy"], x["confidence_threshold"], x["n_candidates"]))

# Сохраняем summary.csv
with output_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"Summary written to {output_csv}")
print(f"Rows: {len(summary_rows)}")
print("Sample:")
for row in summary_rows[:5]:
    print(row)
