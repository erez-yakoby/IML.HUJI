import numpy as np

from IMLearn.metrics.loss_functions import *


def test_misscalssification_error():
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 1, 1, 2])
    c = np.array([1, 1, 2, 3])
    d = np.array([1, 2, 3, 4])
    e = np.array([2, 3, 4, 5])

    assert misclassification_error(a, a) == 0
    assert misclassification_error(a, b) == 1/4
    assert misclassification_error(a, c) == 2/4
    assert misclassification_error(a, d) == 3/4
    assert misclassification_error(a, e) == 4/4

    assert misclassification_error(a, a, False) == 0
    assert misclassification_error(a, b, False) == 1
    assert misclassification_error(a, c, False) == 2
    assert misclassification_error(a, d, False) == 3
    assert misclassification_error(a, e, False) == 4


def test_accuracy():
    true = np.array([1, -1, 1, 1])
    f1 = np.array([1, 1, 1, -1])
    f2 = np.array([1, 1, -1, -1])
    f3 = np.array([1, -1, -1, -1])
    f4 = np.array([-1, -1, -1, -1])

    assert accuracy(true, true) == 1
    assert accuracy(true, f1) == 2/4
    assert accuracy(true, f2) == 1/4
    assert accuracy(true, f3) == 2/4
    assert accuracy(true, f4) == 1/4


def run_tests():
    test_misscalssification_error()
    test_accuracy()

