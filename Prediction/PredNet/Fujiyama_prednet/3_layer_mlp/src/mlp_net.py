"""Network architecture Definition for 3-layer neural net."""
import chainer
import chainer.functions as F
import chainer.links as L

class Regression(chainer.Chain):
    def __init__(self, n_in, n_units, n_out=57600):
        super(Regression, self).__init__(
            l1 = L.Linear(n_in, n_units),
            l2 = L.Linear(n_units, n_units),
            l3 = L.Linear(n_units, n_out)
        )

    def __call__(self, x):
        h1 = F.dropout(F.relu(self.l1(x)))
        h2 = F.dropout(F.relu(self.l2(h1)))
        return self.l3(h2)
