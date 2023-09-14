import numpy as np
import pandas as pd


# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:

    def __init__(self, label=None, children=None):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.label = label
        self.children = {}
        pass

    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """

        if len(y.unique()) == 1:
            self.label = y.iloc[0]
            self.children = None
            return
        print("use information gain")
        self.label = find_split_feature_ig(X, y)
        child_names = X[self.label].unique()
        for child in child_names:
            kept_rows = X[self.label] == child
            subtree = DecisionTree(child)
            subtree.fit(X[kept_rows], y[kept_rows])
            self.children[child] = subtree

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """

        if self.children is None:
            return [self.label] * X.shape[0]
        return pd.Series([self.predict_row(r[1]) for r in X.iterrows()])

    def predict_row(self, row):
        if self.children is None:
            return self.label
        if row[self.label] in self.children:
            return self.children[row[self.label]].predict_row(row.drop(self.label))
        else:
            return "unknown"

    def get_rules(self, previous_rule=None):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        if self.children is None:
            return [(previous_rule, self.label)]
        rules = []
        for feature_value in self.children.keys():
            current_rule = [(self.label, feature_value)]
            if previous_rule is not None:
                current_rule = previous_rule + current_rule
            rules += self.children[feature_value].get_rules(current_rule)
        return rules


# --- Some utility functions

def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true.values == y_pred.values).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


def find_split_feature_ig(samples, results):
    """
    Args:
        samples (pd.Dataframe):
            observed values of the independants variables
        results (pd.Serie :
            observed value of the dependant variable for each sample
    Returns:
        the dataframe column that has the most information gain

    Notes:
        propably can be optimized by using better panda or numpy, here usage of 2 nested for loops
    """
    sample_count = samples.shape[0]
    features_information_gain = pd.Series(index=samples.columns)

    for feature in samples.columns:
        features_information_gain[feature] = 0
        possible_values, count = np.unique(samples[feature], return_counts=True)

        for index, value in enumerate(possible_values):
            results_given_value = results[samples[feature] == value].value_counts()
            value_proba = count[index] / sample_count
            features_information_gain[feature] -= entropy(results_given_value) * value_proba

    features_information_gain = features_information_gain + entropy(results.value_counts())
    return features_information_gain.idxmax()


def find_split_feature_gr(samples, results):
    """
    find the plit feature in the dataset according to the highest gain ratio (idea from 1986 quinlan)
    Args:
        samples (pd.Dataframe):
            observed values of the independants variables
        results (pd.Serie :
            observed value of the dependant variable for each sample
    Returns:
        the dataframe column that has the most information gain

    Notes:
        propably can be optimized by using better panda or numpy, here usage of 2 nested for loops
    """
    sample_count = samples.shape[0]
    features_gain_ratio = pd.Series(index=samples.columns)

    for feature in samples.columns:
        features_gain_ratio[feature] = 0
        feature_split_information = 0
        possible_values, count = np.unique(samples[feature], return_counts=True)

        for index, value in enumerate(possible_values):
            results_given_value = results[samples[feature] == value].value_counts()
            value_proba = count[index] / sample_count
            feature_split_information -= value_proba * np.log2(value_proba)
            features_gain_ratio[feature] -= entropy(results_given_value) * value_proba
        features_gain_ratio[feature] /= feature_split_information
    features_gain_ratio = features_gain_ratio + entropy(results.value_counts())
    features_gain_ratio = features_gain_ratio
    return features_gain_ratio.idxmax()
