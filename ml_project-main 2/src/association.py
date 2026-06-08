import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules




def generate_rules(cluster, min_support=0.02, min_confidence=0.15):
    train_cluster_basket = list(cluster['list_of_goods'])

    te = TransactionEncoder()
    te_fit = te.fit(train_cluster_basket).transform(train_cluster_basket)


    transactions_items_train = pd.DataFrame(te_fit, columns = te.columns_)

    support_itemsets_train = apriori(transactions_items_train, min_support, use_colnames=True)
    support_itemsets_train = support_itemsets_train.sort_values(by='support', ascending=False)


    rules_train = association_rules(support_itemsets_train, metric='confidence', min_threshold = min_confidence)
    rules_train = rules_train.sort_values('lift', ascending=False)

    
    return transactions_items_train, support_itemsets_train, rules_train