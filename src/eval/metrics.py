import numpy as np


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    return (tp + tn) / (tp + tn + fp + fn)


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def f1_score(tp: int, tn: int, fp: int, fn: int) -> float:
    return (2 * precision(tp, fp) * recall(tp, fn)) / (
        precision(tp, fp) + recall(tp, fn)
    )


def confusion_matrix(tp: int, tn: int, fp: int, fn: int) -> np.ndarray:
    return np.array([[tp, fp], [fn, tn]])
