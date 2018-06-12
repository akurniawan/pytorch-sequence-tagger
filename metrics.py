from __future__ import division

from ignite.metrics.metric import Metric
from sklearn.metrics import accuracy_score, classification_report


class SequenceTagAccuracy(Metric):
    def __init__(self, vocabs, output_transform=lambda x: x):
        super(SequenceTagAccuracy, self).__init__(output_transform)
        self._vocabs = vocabs

    def reset(self):
        self._predicted_tags = []
        self._real_tags = []

    def update(self, output):
        y_pred, y = output
        current_pred = y_pred.numpy().tolist()
        current_label = y.numpy().tolist()
        for pred, label in zip(current_label, current_pred):
            self._predicted_tags += [self._vocabs.itos[p] for p in pred]
            self._real_tags += [self._vocabs.itos[l] for l in label]

    def compute(self):
        accuracy = accuracy_score(self._real_tags, self._predicted_tags)
        print(classification_report(self._real_tags, self._predicted_tags))

        return accuracy
