from collections import defaultdict


def min_groups_per_class(labels, groups):
    """Menor número de vídeos/intérpretes distintos entre as classes.

    Usado para decidir se um split por grupo (StratifiedGroupKFold) é
    possível sem deixar alguma classe de fora de treino ou teste.
    """
    groups_by_class = defaultdict(set)
    for label, group in zip(labels, groups):
        groups_by_class[label].add(group)

    return min((len(g) for g in groups_by_class.values()), default=0)
