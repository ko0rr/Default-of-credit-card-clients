
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# 1. Load predictions

df = pd.read_csv(RESULTS / "test_predictions.csv")

required = {
    "ID", "actual_default", "predicted_pd", "LIMIT_BAL"
}
assert required.issubset(df.columns)
assert df["ID"].is_unique
assert df["predicted_pd"].between(0, 1).all()

print("Clients:", len(df))
print("Overall default rate:", round(df["actual_default"].mean(), 4))


# 2. Compare risk strategies

strategies = {
    "Conservative": 0.10,
    "Balanced": 0.15,
    "Aggressive": 0.20,
}

rows = []

for name, threshold in strategies.items():
    selected = df[df["predicted_pd"] <= threshold]

    rows.append({
        "strategy": name,
        "pd_threshold": threshold,
        "selected_clients": len(selected),
        "coverage": len(selected) / len(df),
        "avg_predicted_pd": selected["predicted_pd"].mean(),
        "observed_default_rate": selected["actual_default"].mean(),
        "avg_credit_limit": selected["LIMIT_BAL"].mean(),
    })

strategy_df = pd.DataFrame(rows)

print("\nStrategy comparison:")
print(strategy_df.to_string(index=False))

strategy_df.to_csv(
    RESULTS / "strategy_comparison.csv",
    index=False,
)


# 3. Visualize coverage vs risk

plt.figure(figsize=(8, 5))

for _, row in strategy_df.iterrows():
    plt.scatter(
        row["coverage"] * 100,
        row["avg_predicted_pd"] * 100,
        s=110,
    )
    plt.annotate(
        row["strategy"],
        (
            row["coverage"] * 100,
            row["avg_predicted_pd"] * 100,
        ),
        xytext=(5, 7),
        textcoords="offset points",
    )

plt.xlabel("Selected clients (%)")
plt.ylabel("Average predicted PD (%)")
plt.title("Coverage vs Credit Risk")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    FIGURES / "09_coverage_vs_risk.png",
    dpi=160,
)
plt.close()


# 4. Expected Loss scenario

# Hypothetical assumptions, NOT observed bank parameters
LIMIT_INCREASE = 20_000      # NTD per selected client
ADDITIONAL_DRAWDOWN = 0.50   # Share of added limit drawn
LGD = 0.60                   # Loss Given Default

# Proxy for additional outstanding exposure
incremental_ead = LIMIT_INCREASE * ADDITIONAL_DRAWDOWN

loss_rows = []

for name, threshold in strategies.items():
    selected = df[df["predicted_pd"] <= threshold]

    # Conditional estimate under fixed historical PD
    expected_loss = (
        selected["predicted_pd"]
        * LGD
        * incremental_ead
    ).sum()

    loss_rows.append({
        "strategy": name,
        "selected_clients": len(selected),
        "incremental_ead_per_client": incremental_ead,
        "expected_loss_ntd": expected_loss,
        "expected_loss_per_client": (
            expected_loss / len(selected)
            if len(selected) else np.nan
        ),
    })

loss_df = pd.DataFrame(loss_rows)

print("\nExpected loss scenario:")
print(loss_df.to_string(index=False))

loss_df.to_csv(
    RESULTS / "expected_loss_scenario.csv",
    index=False,
)


# 5. Sensitivity analysis

# Test how assumptions change the estimated losses
lgd_values = [0.4, 0.6, 0.8]
drawdown_values = [0.25, 0.5, 0.75]

sensitivity_rows = []

for name, threshold in strategies.items():
    selected = df[df["predicted_pd"] <= threshold]

    for lgd in lgd_values:
        for drawdown in drawdown_values:
            ead = LIMIT_INCREASE * drawdown

            expected_loss = (
                selected["predicted_pd"] * lgd * ead
            ).sum()

            sensitivity_rows.append({
                "strategy": name,
                "lgd": lgd,
                "drawdown": drawdown,
                "expected_loss_ntd": expected_loss,
            })

sensitivity = pd.DataFrame(sensitivity_rows)

sensitivity.to_csv(
    RESULTS / "loss_sensitivity.csv",
    index=False,
)


# 6. Approximate A/B test sample size

def ab_sample_size(p_control, p_treatment,
                   alpha=0.05, power=0.80):
    """
    Approximate sample size per group for two
    independent proportions, with equal allocation.
    """
    p_avg = (p_control + p_treatment) / 2

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
        + z_beta * np.sqrt(
            p_control * (1 - p_control)
            + p_treatment * (1 - p_treatment)
        )
    ) ** 2

    denominator = (p_treatment - p_control) ** 2

    return int(np.ceil(numerator / denominator))


# Hypothetical activation rates
p_control = 0.20
p_treatment = 0.23

n_per_group = ab_sample_size(
    p_control,
    p_treatment,
)

print("\nA/B test planning:")
print("Baseline activation:", p_control)
print("Expected activation:", p_treatment)
print("Absolute MDE:", p_treatment - p_control)
print("Sample size per group:", n_per_group)
print("Total sample size:", 2 * n_per_group)

pd.DataFrame([{
    "control_activation": p_control,
    "treatment_activation": p_treatment,
    "absolute_mde": p_treatment - p_control,
    "alpha": 0.05,
    "power": 0.80,
    "sample_per_group": n_per_group,
    "total_sample": 2 * n_per_group,
}]).to_csv(
    RESULTS / "ab_test_design.csv",
    index=False,
)

