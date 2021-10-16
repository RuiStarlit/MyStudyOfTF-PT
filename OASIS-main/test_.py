# -*- coding: UTF-8 -*-
"""
written by Rui
file:test_.py
create time:2021/04/10
"""
import numpy as np
from Oasis import Oasis
from sklearn import datasets
import matplotlib.pyplot as plt

if __name__ == "__main__":

    digits = datasets.load_digits()

    X_train = digits.data[500:] / 16
    X_test = digits.data[:500] / 16
    y_train = digits.target[500:]
    y_test = digits.target[:500]

    model = Oasis(n_iter=100000, do_psd=True, psd_every=3,
                  save_path="/tmp/oasis_test").fit(X_train, y_train,
                                                   verbose=True)

    errrate = model.predict(X_test, X_train, y_test, y_train, maxk=1000)
    print("Min error rate: %6.4f at k=%d" \
          % (min(errrate), np.argmin(errrate) + 1))
    plt.figure()
    plt.plot(errrate)
    plt.show()

    n_features = X_train.shape[1]
    W = model._W.view()
    W.shape = (n_features, n_features)

    print(W[0:5, 0:5])
    model = Oasis()
    print(model.n_iter)