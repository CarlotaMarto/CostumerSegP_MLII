import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import ast
from IPython.display import display


def read_basket(filepath="customer_basket.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
def read_id_and_cluster(filepath="id_and_cluster.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
def read_makrp_clu(filepath="id_and_cluster.csv"):
    """
    Reads customer data from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'customer_info.csv' in the current directory.

    Returns
    -------
    pd.DataFrame
        The loaded customer information DataFrame.
    """
    try:
        customer_info = pd.read_csv(filepath)
        return customer_info
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()

def cluster_association_rules(cluster_df):
    cluster_df = cluster_df.drop(["customer_id", "Unnamed: 0", "Cluster"], axis=1)
    train = cluster_df[:int(len(cluster_df)*0.8)]
    test = cluster_df[int(len(cluster_df)*0.8):]
    print('We have {} rows in the train set.'.format(len(train)))
    print('We have {} rows in the test set.'.format(len(test)))
    transactions_train = train['list_of_goods'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    te = TransactionEncoder()
    te_fit = te.fit(transactions_train).transform(transactions_train)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)
    frequent_itemsets_grocery = apriori(
    transactions_items, min_support=0.05, use_colnames=True
    )
    print("\n Frequent Itemsets (support >= 0.5)")
    display(frequent_itemsets_grocery.sort_values(by='support', ascending=False))
    rules_grocery = association_rules(frequent_itemsets_grocery,
                                  metric="confidence",
                                  min_threshold=0.2, num_itemsets=len(frequent_itemsets_grocery))
    print("\n Association rules (confidence >= 0.2)")
    display(rules_grocery)

    frequent_itemsets_grocery_iter_2 = apriori(
    transactions_items, min_support=0.02, use_colnames=True
    )

    rules_grocery_iter_2 = association_rules(frequent_itemsets_grocery_iter_2,
                                  metric="confidence",
                                  min_threshold=0.2, num_itemsets=len(frequent_itemsets_grocery_iter_2))
    print("\n Top 10 rules by lift (support >= 0.02)")
    display(rules_grocery_iter_2.sort_values(by='lift', ascending=False).head(10))
    train_data_rules = rules_grocery_iter_2[['antecedents','consequents','lift']]
    transactions_test = test['list_of_goods'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).tolist()
    test_fit = te.transform(transactions_test)
    transactions_items_test = pd.DataFrame(test_fit, columns=te.columns_)
    frequent_itemsets_test = apriori(
        transactions_items_test, min_support=0.05, use_colnames=True
        )
    rules_grocery_test = association_rules(frequent_itemsets_test,
                                      metric="confidence",
                                      min_threshold=0.2, num_itemsets=len(frequent_itemsets_test))
    print("\n Test set: top 10 rules by lift (support >= 0.05)")
    display(rules_grocery_test.sort_values(by='lift', ascending=False).head(10))
    train_data_rules = rules_grocery_iter_2[['antecedents','consequents','lift']].copy()
    train_data_rules['rule'] = train_data_rules['antecedents'].apply(sorted).astype(str) + '->' + train_data_rules['consequents'].apply(sorted).astype(str)
    rules_grocery_test['rule'] = rules_grocery_test['antecedents'].apply(sorted).astype(str) + '->' + rules_grocery_test['consequents'].apply(sorted).astype(str)
    train_data_rules['rule'] = train_data_rules['antecedents'].apply(sorted).astype(str) + '->' + train_data_rules['consequents'].apply(sorted).astype(str)

    eval_df = train_data_rules[['rule', 'lift']].merge(
        rules_grocery_test[['rule', 'lift']], on='rule', suffixes=('_train', '_test')
    )
    mean_diff = ((eval_df['lift_train'] - eval_df['lift_test']).abs() / eval_df['lift_train']).mean()
    print(f"Mean Relative Lift Difference: {mean_diff}")