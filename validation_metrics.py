import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

class Metrics():
    def __init__(self, predictions, test_labels):
        self.predictions = predictions
        self.test_labels = test_labels
        self.cfn_matrix = self._get_confusion_matrix()

    def _get_confusion_matrix(self):
        return confusion_matrix(self.test_labels, self.predictions)

    def generate_confusion_matrix(self):
        # Confusion matrix displaying counts
        sns.heatmap(self.cfn_matrix, annot=True)
        # Confusion matrix displaying percentages
        sns.heatmap(self.cfn_matrix / np.sum(cfn_matrix), annot=True, fmt='.2%', cmap='Blues')
        plt.show()

    def generate_report(self):
        score = accuracy_score(self.test_labels, self.predictions)
        print("[INFO]: Classifier achieved a {}% accuracy.".format(score))
        report = clssification_report(self.test_labels, self.predictions)
        print("[INFO]: Generating classifier stats.")
        print(report)


def build_metrics(predictions, test_labels):
    return Metrics(predictions, test_labels)
