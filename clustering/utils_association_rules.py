"""Utilities for association rule mining per customer segment."""

import ast
import random
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


PROJECT_PALETTE = [
    "#B87540",
    "#B2543D",
    "#7E6A43",
    "#A8B7BA",
    "#D8C0B4",
    "#C8AB8C",
    "#5A3516",
    "#B98F70",
]
MAIN_COLOR = "#B87540"
ACCENT_COLOR = "#B2543D"
NOTE_COLOR = "#7E6A43"
SECONDARY_COLOR = "#A8B7BA"
LIGHT_COLOR = "#F3EEE6"
DARK_COLOR = "#5A3516"


def build_onehot(transactions):
    """Encode a list of transaction item-lists as a boolean one-hot DataFrame."""
    te = TransactionEncoder()
    matrix = te.fit_transform(transactions)
    return pd.DataFrame(matrix, columns=te.columns_)


def mine_rules(
    transactions,
    min_support=0.01,
    min_confidence=0.20,
    min_lift=1.2,
):
    """Run apriori on a list of transactions and return filtered association rules."""
    onehot = build_onehot(transactions)
    frequent = apriori(onehot, min_support=min_support, use_colnames=True)
    if frequent.empty:
        return pd.DataFrame(), pd.DataFrame()

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] >= min_lift].copy()
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules, frequent


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



def rule_overlap_summary(all_rules, cluster_names, n=10):
    """Summarise how distinct the top rules are across segments."""
    rule_clusters = {}
    consequent_clusters = {}
    top_by_cluster = {}

    for cluster_id, rules in all_rules.items():
        top = rules.head(n).copy()
        top_by_cluster[cluster_id] = top
        for _, row in top.iterrows():
            r_key = rule_key(row)
            c_key = ", ".join(sorted(row["consequents"]))
            rule_clusters.setdefault(r_key, set()).add(cluster_id)
            consequent_clusters.setdefault(c_key, set()).add(cluster_id)

    rows = []
    for cluster_id, top in top_by_cluster.items():
        if top.empty:
            rows.append({
                "cluster": cluster_id,
                "segment": cluster_names.get(cluster_id, str(cluster_id)),
                "top_rules_checked": 0,
                "unique_consequents": 0,
                "repeated_exact_rules": 0,
                "repeated_consequents": 0,
                "main_consequents": "",
                "interpretation": "no rules available",
            })
            continue

        rule_keys = top.apply(rule_key, axis=1)
        consequent_keys = top["consequents"].apply(lambda x: ", ".join(sorted(x)))
        repeated_rules = sum(len(rule_clusters[k]) > 1 for k in rule_keys)
        repeated_cons = sum(len(consequent_clusters[k]) > 1 for k in consequent_keys)
        main_cons = "; ".join(consequent_keys.value_counts().head(3).index.tolist())

        if repeated_rules >= max(1, len(top) * 0.5):
            interpretation = "mostly generic"
        elif repeated_cons >= max(1, len(top) * 0.5):
            interpretation = "shared product theme"
        else:
            interpretation = "more segment specific"

        rows.append({
            "cluster": cluster_id,
            "segment": cluster_names.get(cluster_id, str(cluster_id)),
            "top_rules_checked": len(top),
            "unique_consequents": consequent_keys.nunique(),
            "repeated_exact_rules": int(repeated_rules),
            "repeated_consequents": int(repeated_cons),
            "main_consequents": main_cons,
            "interpretation": interpretation,
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


def build_campaign_table(all_rules, cluster_names, n=3, excluded_recommendations=None, scan_limit=50):
    """Build a campaign table, optionally skipping unsuitable recommendations."""
    excluded_recommendations = excluded_recommendations or {}
    rows = []
    for cluster_id, rules in all_rules.items():
        blocked = {item.lower() for item in excluded_recommendations.get(cluster_id, [])}
        selected = 0
        for _, row in rules.head(scan_limit).iterrows():
            consequents = sorted(row["consequents"])
            if any(item.lower() in blocked for item in consequents):
                continue
            rows.append({
                "cluster": cluster_id,
                "segment": cluster_names.get(cluster_id, str(cluster_id)),
                "if_buys": ", ".join(sorted(row["antecedents"])),
                "promote": ", ".join(consequents),
                "confidence": round(row["confidence"], 2),
                "lift": round(row["lift"], 2),
            })
            selected += 1
            if selected >= n:
                break
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
        axes[idx].barh(top["rule"][::-1], top["lift"][::-1], color=MAIN_COLOR)
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


def basket_coverage_by_segment(data_dir, cluster_names):
    """Check how much of each segment is represented in the basket data."""
    data_dir = str(data_dir)
    basket = pd.read_csv(f"{data_dir}/customer_basket.csv")
    segments = pd.read_csv(f"{data_dir}/customer_segments.csv")

    basket_customers = basket["customer_id"].dropna().unique()
    rows = []
    for cluster_id in sorted(segments["cluster"].unique()):
        segment_ids = segments.loc[segments["cluster"] == cluster_id, "customer_id"]
        in_basket = segment_ids.isin(basket_customers)
        rows.append({
            "cluster": cluster_id,
            "segment": cluster_names.get(cluster_id, str(cluster_id)),
            "segment_customers": int(segment_ids.nunique()),
            "customers_with_basket": int(in_basket.sum()),
            "coverage_%": round(float(in_basket.mean() * 100), 2),
            "transactions": int(basket[basket["customer_id"].isin(segment_ids)]["invoice_id"].nunique())
            if "invoice_id" in basket.columns else int(basket["customer_id"].isin(segment_ids).sum()),
        })

    out = pd.DataFrame(rows)
    print(f"Basket rows: {len(basket):,}")
    print(f"Customers in final segmentation: {segments['customer_id'].nunique():,}")
    print(f"Segmented customers with basket data: {segments['customer_id'].isin(basket_customers).sum():,}")
    return out


def basket_quality_summary(data_dir, top_n=15):
    """Summarise basket uniqueness and the most common items."""
    data_dir = str(data_dir)
    basket = pd.read_csv(f"{data_dir}/customer_basket.csv")

    summary = {
        "basket_rows": len(basket),
        "unique_invoices": basket["invoice_id"].nunique() if "invoice_id" in basket else None,
        "duplicated_invoice_rows": int(basket["invoice_id"].duplicated().sum()) if "invoice_id" in basket else None,
        "unique_customers": basket["customer_id"].nunique() if "customer_id" in basket else None,
        "duplicated_invoice_customer_rows": int(basket[["invoice_id", "customer_id"]].duplicated().sum())
        if {"invoice_id", "customer_id"}.issubset(basket.columns) else None,
    }

    items = basket["list_of_goods"].apply(ast.literal_eval).tolist()
    onehot = build_onehot(items)
    item_support = (
        onehot.mean()
        .sort_values(ascending=False)
        .head(top_n)
        .mul(100)
        .round(2)
        .rename_axis("item")
        .reset_index(name="transaction_support_%")
    )

    return pd.DataFrame([summary]), item_support


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


_CAMPAIGN_TEMPLATES = [
    "Buy {antecedents} and get {discount}% off {consequents}!",
    "Exclusive deal: purchase {antecedents} and receive {consequents} at {discount}% off.",
    "Buy {antecedents} and get {discount}% off {consequents} on your next purchase.",
    "This week only: {antecedents} + {consequents} together for just {bundle_pct}% of their combined price.",
    "Love {antecedents}? You'll love {consequents} too — {discount}% off just for you.",
]


def _pick_template(lift, confidence, rng):
    if lift >= 3.0:
        return _CAMPAIGN_TEMPLATES[0]
    if confidence >= 0.6:
        return _CAMPAIGN_TEMPLATES[1]
    return random.choice(_CAMPAIGN_TEMPLATES[2:])


def _discount_from_lift(lift):
    if lift >= 4.0:
        return 30
    if lift >= 3.0:
        return 25
    if lift >= 2.0:
        return 20
    return 15


def format_campaign_creative(campaigns, cluster_names=None, random_state=42):
    """Convert the campaign table into ready-to-use creative campaign texts with discount levels."""
    random.seed(random_state)

    out = campaigns.copy()

    if cluster_names is not None:
        out["segment"] = out["cluster"].map(cluster_names).fillna(out["segment"])

    texts, discounts, rationales = [], [], []

    for _, row in out.iterrows():
        ant = row["if_buys"].title()
        con = row["promote"].title()
        lift = float(row["lift"])
        conf = float(row["confidence"])
        disc = _discount_from_lift(lift)
        bundle_pct = round((1 - disc / 100) * 100)

        template = _pick_template(lift, conf, random)
        text = template.format(
            antecedents=ant,
            consequents=con,
            discount=disc,
            bundle_pct=bundle_pct,
        )
        rationale = (
            f"Rule lift {lift:.2f} — customers who buy {ant} are {lift:.1f}× more "
            f"likely than average to also buy {con} "
            f"(confidence {conf:.0%})."
        )
        texts.append(text)
        discounts.append(disc)
        rationales.append(rationale)

    out["discount_%"] = discounts
    out["campaign_text"] = texts
    out["campaign_rationale"] = rationales
    return out


def print_campaign_report(creative_campaigns):
    """Pretty-print the creative campaign report grouped by segment."""
    for segment, group in creative_campaigns.groupby("segment"):
        print(f"\n{'=' * 60}")
        print(f"  Segment: {segment}")
        print(f"{'=' * 60}")
        for i, (_, row) in enumerate(group.iterrows(), 1):
            print(f"\n  Campaign {i}:")
            print(f"  ➤  {row['campaign_text']}")
            print(f"     ({row['campaign_rationale']})")
    print()


def top_rule_per_segment(all_rules, cluster_names):
    """Return a summary table with the single strongest rule (by lift) per segment."""
    rows = []
    for cluster_id, rules in all_rules.items():
        if rules.empty:
            continue
        top = rules.nlargest(1, "lift").iloc[0]
        rows.append({
            "segment": cluster_names.get(cluster_id, str(cluster_id)),
            "if_buys": ", ".join(sorted(top["antecedents"])),
            "then_buys": ", ".join(sorted(top["consequents"])),
            "support": round(top["support"], 3),
            "confidence": round(top["confidence"], 3),
            "lift": round(top["lift"], 3),
        })
    return pd.DataFrame(rows).set_index("segment")

