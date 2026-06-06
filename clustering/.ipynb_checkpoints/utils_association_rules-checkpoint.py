"""Utilities for association rule mining per customer segment."""

import ast
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
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


def rule_key(row):
    """Create a stable text key for an association rule."""
    left = ", ".join(sorted(row["antecedents"]))
    right = ", ".join(sorted(row["consequents"]))
    return f"{left} -> {right}"


def compare_train_test_rules(
    df,
    cluster_names,
    min_support=0.02,
    min_confidence=0.30,
    min_lift=1.2,
    test_size=0.20,
    random_state=42,
):
    """Compare rule lift stability between train and test transactions."""
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=random_state).index
    df_shuffled = df.iloc[rng].reset_index(drop=True)
    rows = []

    for cluster_id in sorted(df_shuffled["cluster"].unique()):
        segment_df = df_shuffled[df_shuffled["cluster"] == cluster_id].reset_index(drop=True)
        split = int(len(segment_df) * (1 - test_size))
        train_tx = segment_df.loc[:split - 1, "items"].tolist()
        test_tx = segment_df.loc[split:, "items"].tolist()

        train_rules, _ = mine_rules(train_tx, min_support, min_confidence, min_lift)
        test_rules, _ = mine_rules(test_tx, min_support, min_confidence, min_lift)

        if train_rules.empty or test_rules.empty:
            rows.append({
                "cluster": cluster_id,
                "segment": cluster_names.get(cluster_id, str(cluster_id)),
                "train_rules": len(train_rules),
                "test_rules": len(test_rules),
                "matched_rules": 0,
                "mean_lift_difference_%": None,
            })
            continue

        train_eval = train_rules.copy()
        test_eval = test_rules.copy()
        train_eval["rule"] = train_eval.apply(rule_key, axis=1)
        test_eval["rule"] = test_eval.apply(rule_key, axis=1)
        matched = train_eval[["rule", "lift"]].merge(
            test_eval[["rule", "lift"]],
            on="rule",
            suffixes=("_train", "_test"),
        )

        if matched.empty:
            mean_diff = None
        else:
            rel_diff = (
                (matched["lift_train"] - matched["lift_test"]).abs()
                / matched["lift_train"].replace(0, pd.NA)
            )
            mean_diff = round(float(rel_diff.mean() * 100), 2)

        rows.append({
            "cluster": cluster_id,
            "segment": cluster_names.get(cluster_id, str(cluster_id)),
            "train_rules": len(train_rules),
            "test_rules": len(test_rules),
            "matched_rules": len(matched),
            "mean_lift_difference_%": mean_diff,
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
        axes[idx].barh(top["rule"][::-1], top["lift"][::-1], color="#B87540")
        axes[idx].set_title(f"Cluster {cluster_id}: {name}", fontsize=9)
        axes[idx].set_xlabel("Lift")
        axes[idx].tick_params(axis="y", labelsize=7)

    for ax in axes[n_clusters:]:
        ax.axis("off")

    plt.suptitle("Top association rules by lift per segment", y=1.01, fontsize=12)
    plt.tight_layout()
    plt.show()



def load_segment_transactions(data_dir, cluster_names=None):
    """Load baskets, attach clusters and parse basket item lists."""
    data_dir = str(data_dir)
    basket = pd.read_csv(f"{data_dir}/customer_basket.csv")
    segments = pd.read_csv(f"{data_dir}/customer_segments.csv")

    basket["items"] = basket["list_of_goods"].apply(ast.literal_eval)
    df = basket.merge(segments[["customer_id", "cluster"]], on="customer_id", how="inner")
    if cluster_names is not None:
        df["cluster_name"] = df["cluster"].map(cluster_names)
    return df


def transaction_overview(df):
    """Print a compact overview of the transaction data used for rules."""
    print(f"Transactions: {len(df):,}")
    print(f"Customers with cluster: {df['customer_id'].nunique():,}")
    print(f"Unique items: {df['items'].explode().nunique()}")
    print()
    print("Transactions per cluster:")
    display(df["cluster"].value_counts().sort_index().rename("transactions").to_frame())


def transactions_by_cluster(df):
    """Return transaction item lists by cluster."""
    return {
        c: df.loc[df["cluster"] == c, "items"].tolist()
        for c in sorted(df["cluster"].unique())
    }


def show_transactions_by_cluster(df, cluster_names):
    """Display transaction counts by cluster and segment name."""
    rows = []
    for cluster_id, txns in transactions_by_cluster(df).items():
        rows.append({
            "cluster": cluster_id,
            "segment": cluster_names.get(cluster_id, str(cluster_id)),
            "transactions": len(txns),
        })
    out = pd.DataFrame(rows)
    display(out)
    return out


def display_top_rules(all_rules, cluster_names, n=5):
    """Display top rules per segment in a readable format."""
    for cluster_id, rules in all_rules.items():
        name = cluster_names.get(cluster_id, str(cluster_id))
        print(f"\n=== Cluster {cluster_id}: {name} ===")
        top = rules.head(n)[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        top["antecedents"] = top["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        top["consequents"] = top["consequents"].apply(lambda x: ", ".join(sorted(x)))
        top[["support", "confidence", "lift"]] = top[["support", "confidence", "lift"]].round(3)
        display(top)


def export_campaign_table(campaigns, data_dir, filename="segment_campaign_rules.csv"):
    """Save the campaign recommendation table."""
    output_path = f"{data_dir}/{filename}"
    campaigns.to_csv(output_path, index=False)
    print(f"Saved {filename}")
    print(f"Total campaign suggestions: {len(campaigns)}")
    return output_path

