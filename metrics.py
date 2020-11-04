import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

class Metrics():
    def __init__(self, predictions, test_labels):
        self.predictions = predictions
        self.test_labels = test_labels
        self.cfn_matrix = self._get_confusion_matrix()

    def _get_confusion_matrix(self):
        return confusion_matrix(self.test_labels, self.predictions)

    def print_confusion_matrix(self):
        print(sns.heatmap(self.cfn_matrix, annot=True))

def build_metrics(predictions, test_labels):
    return Metrics(predictions, test_labels)
