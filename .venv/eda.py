from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

for folder in (DATA_DIR, FIGURES_DIR, RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def run_query(connection, name, query):
    result = connection.execute(query).df()
    print(f"\n{name}\n{result.to_string(index=False)}")
    result.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    return result


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()

# Читаемые названия признаков
column_mapping = {
    "X1": "LIMIT_BAL",
    "X2": "SEX",
    "X3": "EDUCATION",
    "X4": "MARRIAGE",
    "X5": "AGE",
    "X6": "PAY_0",
    "X7": "PAY_2",
    "X8": "PAY_3",
    "X9": "PAY_4",
    "X10": "PAY_5",
    "X11": "PAY_6",
    "X12": "BILL_AMT1",
    "X13": "BILL_AMT2",
    "X14": "BILL_AMT3",
    "X15": "BILL_AMT4",
    "X16": "BILL_AMT5",
    "X17": "BILL_AMT6",
    "X18": "PAY_AMT1",
    "X19": "PAY_AMT2",
    "X20": "PAY_AMT3",
    "X21": "PAY_AMT4",
    "X22": "PAY_AMT5",
    "X23": "PAY_AMT6",
}

dataset = fetch_ucirepo(id=350)
df = dataset.data.features.copy()
df["default"] = dataset.data.targets.iloc[:, 0].to_numpy()

# Общая информация о датасете
df = df.rename(columns=column_mapping)
print(df.head())
print(df.columns.tolist())
if "ID" not in df.columns and dataset.data.ids is not None:
    df.insert(0, "ID", dataset.data.ids.iloc[:, 0].to_numpy())

print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nStatistics:\n", df.describe())


# Качество датасета
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

if "ID" in df.columns:
    print("Duplicate IDs:", df["ID"].duplicated().sum())

for col in ["EDUCATION", "MARRIAGE", "PAY_0"]:
    print(f"\n{col}:\n", df[col].value_counts().sort_index())

assert set(df["default"].dropna().unique()).issubset({0, 1})
assert (df["LIMIT_BAL"] > 0).all()

print(f"\nOverall default rate: {df['default'].mean():.2%}")


# EDA
df["limit_group"] = pd.cut(
    df["LIMIT_BAL"],
    bins=[0, 50_000, 100_000, 200_000, 500_000, np.inf],
    labels=["<=50k", "50-100k", "100-200k", "200-500k", ">500k"],
    include_lowest=True
)

df["sep_late"] = (df["PAY_0"] >= 1).astype(int)

df["bill_to_limit"] = df["BILL_AMT1"] / df["LIMIT_BAL"]

df["bill_ratio_group"] = pd.cut(
    df["bill_to_limit"],
    bins=[-np.inf, 0, 0.25, 0.5, 0.75, 1, np.inf],
    labels=["<=0", "0-25%", "25-50%", "50-75%", "75-100%", ">100%"]
)

df.to_csv(DATA_DIR / "clients_prepared.csv", index=False)


# SQL запросы к датасету
con = duckdb.connect(database=":memory:")
con.register("clients", df)

overall = run_query(con, "overall", """
    SELECT
        COUNT(*) AS clients,
        SUM("default") AS defaults,
        ROUND(100.0 * AVG("default"), 2) AS default_rate_pct
    FROM clients
""")

limits = run_query(con, "by_limit", """
    SELECT
        CAST(limit_group AS VARCHAR) AS limit_group,
        COUNT(*) AS clients,
        ROUND(AVG(LIMIT_BAL), 0) AS avg_limit_ntd,
        ROUND(100.0 * AVG("default"), 2) AS default_rate_pct
    FROM clients
    GROUP BY limit_group
    ORDER BY MIN(LIMIT_BAL)
""")

repayment = run_query(con, "by_repayment_status", """
    SELECT
        PAY_0 AS repayment_status,
        COUNT(*) AS clients,
        ROUND(100.0 * AVG("default"), 2) AS default_rate_pct
    FROM clients
    GROUP BY PAY_0
    ORDER BY PAY_0
""")

cross = run_query(con, "by_limit_and_delay", """
    SELECT
        CAST(limit_group AS VARCHAR) AS limit_group,
        sep_late,
        COUNT(*) AS clients,
        ROUND(100.0 * AVG("default"), 2) AS default_rate_pct
    FROM clients
    GROUP BY limit_group, sep_late
    ORDER BY MIN(LIMIT_BAL), sep_late
""")

delay_share = run_query(con, "delay_share_by_limit", """
    SELECT
        CAST(limit_group AS VARCHAR) AS limit_group,
        COUNT(*) AS clients,
        ROUND(100.0 * AVG(sep_late), 2) AS delay_share_pct
    FROM clients
    GROUP BY limit_group
    ORDER BY MIN(LIMIT_BAL)
""")

ratios = run_query(con, "by_statement_ratio", """
    SELECT
        CAST(bill_ratio_group AS VARCHAR) AS bill_ratio_group,
        COUNT(*) AS clients,
        ROUND(100.0 * AVG("default"), 2) AS default_rate_pct
    FROM clients
    GROUP BY bill_ratio_group
    ORDER BY MIN(bill_to_limit)
""")


# Визуализация данных
plt.figure(figsize=(8, 4.5))
sns.histplot(df["LIMIT_BAL"] / 1000, bins=40)
plt.xlabel("Credit limit (thousand NTD)")
plt.ylabel("Clients")
plt.title("Credit limit distribution")
save_plot("01_credit_limit_distribution.png")

plt.figure(figsize=(8, 4.5))
sns.barplot(data=limits, x="limit_group", y="default_rate_pct")
plt.xlabel("Credit limit group")
plt.ylabel("Default rate (%)")
plt.title("Default rate by credit limit")
save_plot("02_default_by_limit.png")

plt.figure(figsize=(8, 4.5))
sns.barplot(data=repayment, x="repayment_status", y="default_rate_pct")
plt.xlabel("September repayment status")
plt.ylabel("Default rate (%)")
plt.title("Default rate by repayment status")
save_plot("03_default_by_repayment_status.png")

plt.figure(figsize=(8, 4.5))
sns.barplot(data=ratios, x="bill_ratio_group", y="default_rate_pct")
plt.xlabel("Statement amount / credit limit")
plt.ylabel("Default rate (%)")
plt.title("Default rate by statement-to-limit ratio")
save_plot("04_default_by_statement_ratio.png")

con.close()
