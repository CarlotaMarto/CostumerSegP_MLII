import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def dataframe_to_transaction_items(dataframe_: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a transaction items dataframe

    Transform the column list_of_goods column of a dataframe
    into a transaction items dataframe

    Arguments:
        ----------
         - dataframe_(pd.DataFrame): dataframe with a column 
         name list_of_goods (has inside a list of string values)

    Returns:
        ----------
         -transactions_items(pd.DataFrame): dataframe in the form of
         transaction items of True and False
    '''
    matrix_products = dataframe_.apply(lambda row: row['list_of_goods'
                                                       ].replace(" ", ''
                                                        ).replace("'", ''
                                                        ).replace("[", ''
                                                        ).replace("]", ''
                                                        ).split(","
                                                        ), axis=1).values
    te = TransactionEncoder()
    te_fit = te.fit(matrix_products).transform(matrix_products)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)

    return transactions_items


def creates_associations_rules(transaction_items: pd.DataFrame,
                               min_support: float = 0.05,
                               min_confidence: float = 0.1,
                               max_rules: int = 10) -> pd.DataFrame:
    '''
    Creates a association rules dataframe

    Arguments:
        ----------
         - transaction_items(pd.DataFrame): dataframe with true and
         falses.
         -min_support(float): minimum of support to consider.
         -min_confidence(float): minimum confidence to consider.
         -max_rules(int): number of rules to display.

    Returns:
        ----------
         -rules_items(pd.DataFrame): dataframe with the best associations
         rules based on lift (more lift on top).
    '''
    # Calculate the support
    frequent_itemsets = apriori(transaction_items,
                                min_support=min_support,
                                use_colnames=True)

    # Dataframe of antecendt and consequence
    rules_items = association_rules(frequent_itemsets,
                                    metric="confidence",
                                    min_threshold=min_confidence
                                    ).sort_values(by='lift',
                                                  ascending=False
                                                  ).head(max_rules)

    return rules_items
