"""Utilities for association rule mining per customer segment."""

import warnings
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def build_onehot(transactions):
    """Encode a list of transaction item-lists as a boolean one-hot DataFrame."""
    te = TransactionEncoder()
    matrix = te.fit_transform(transactions)
    return pd.DataFrame(matrix, columns=te.columns_)


def mine_rules(
    transactions,
    min_support=0.02,
    min_confidence=0.30,
    min_lift=1.2,
):
    """Run apriori on a list of transactions and return filtered association rules.

    Falls back to looser thresholds (support=0.01, confidence=0.20, lift>=1.0)
    when fewer than 3 rules are found with the primary parameters.

    Returns a DataFrame sorted by lift descending, or an empty DataFrame.
    """
    onehot = build_onehot(transactions)

    for sup, conf, lift_th in [
        (min_support, min_confidence, min_lift),
        (0.01, 0.20, 1.0),
    ]:
        frequent = apriori(onehot, min_support=sup, use_colnames=True)
        if frequent.empty:
            continue
        rules = association_rules(frequent, metric="confidence", min_threshold=conf)
        rules = rules[rules["lift"] >= lift_th].copy()
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        if len(rules) >= 3:
            if sup != min_support:
                warnings.warn(
                    f"Relaxed thresholds used: support={sup}, confidence={conf}, lift>={lift_th}"
                )
            return rules, frequent

    return pd.DataFrame(), pd.DataFrame()


def mine_rules_per_segment(
    df,
    cluster_names,
    min_support=0.02,
    min_confidence=0.30,
    min_lift=1.2,
):
    """Mine association rules for every cluster in df.

    Parameters
    ----------
    df : DataFrame with columns ['items', 'cluster'] where 'items' is a list of strings
    cluster_names : dict mapping cluster id to segment name

    Returns
    -------
    dict mapping cluster_id -> rules DataFrame
    """
    all_rules = {}
    for cluster_id in sorted(df["cluster"].unique()):
        name = cluster_names.get(cluster_id, str(cluster_id))
        transactions = df.loc[df["cluster"] == cluster_id, "items"].tolist()
        print(f"\n--- Cluster {cluster_id}: {name} ({len(transactions):,} transactions) ---")

        rules, frequent = mine_rules(
            transactions, min_support, min_confidence, min_lift
        )

        if frequent.empty:
            print("  No frequent itemsets found.")
            continue

        print(f"  Frequent itemsets: {len(frequent):,}")
        print(f"  Rules returned: {len(rules):,}")
        all_rules[cluster_id] = rules

    return all_rules


def top_rules_table(all_rules, cluster_names, n=5):
    """Return a tidy DataFrame of the top n rules per cluster."""
    rows = []
    for cluster_id, rules in all_rules.items():
        for _, row in rules.head(n).iterrows():
            rows.append({
                "cluster": cluster_id,
                "segment": cluster_names.get(cluster_id, str(cluster_id)),
                "antecedents": ", ".join(sorted(row["antecedents"])),
                "consequents": ", ".join(sorted(row["consequents"])),
                "support": round(row["support"], 3),
                "confidence": round(row["confidence"], 2),
                "lift": round(row["lift"], 2),
            })
    return pd.DataFrame(rows)


def build_campaign_table(all_rules, cluster_names, n=3):
    """Build a campaign suggestion table from the top n rules per cluster."""
    rows = []
    for cluster_id, rules in all_rules.items():
        for _, row in rules.head(n).iterrows():
            rows.append({
                "cluster": cluster_id,
                "segment": cluster_names.get(cluster_id, str(cluster_id)),
                "if_buys": ", ".join(sorted(row["antecedents"])),
                "promote": ", ".join(sorted(row["consequents"])),
                "confidence": round(row["confidence"], 2),
                "lift": round(row["lift"], 2),
            })
    return pd.DataFrame(rows)


def plot_rules_by_segment(all_rules, cluster_names, n=10):
    """Horizontal bar charts of top rules by lift for each segment."""
    n_clusters = len(all_rules)
    n_cols = 4
    n_rows = -(-n_clusters // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.ravel()

    for idx, (cluster_id, rules) in enumerate(all_rules.items()):
        name = cluster_names.get(cluster_id, str(cluster_id))
        top = rules.head(n).copy()
        top["rule"] = (
            top["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            + " → "
            + top["consequents"].apply(lambda x: ", ".join(sorted(x)))
        )
        axes[idx].barh(top["rule"][::-1], top["lift"][::-1], color="#1B4F72")
        axes[idx].set_title(f"Cluster {cluster_id}: {name}", fontsize=9)
        axes[idx].set_xlabel("Lift")
        axes[idx].tick_params(axis="y", labelsize=7)

    for ax in axes[n_clusters:]:
        ax.axis("off")

    plt.suptitle("Top association rules by lift per segment", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.show()
