from sklearn.model_selection import GroupKFold


def group_splits(groups, n_splits: int = 5):
    return GroupKFold(n_splits=n_splits).split(range(len(groups)), groups=groups)
