# coding: utf-8
import pandas as pd
import numpy as np
from sklearn import metrics


def classification():
    cls_data = pd.read_csv('classification.csv')
    TP = np.sum((cls_data['pred'] == 1) & (cls_data['true'] == 1))
    FP = np.sum((cls_data['pred'] == 1) & (cls_data['true'] == 0))
    FN = np.sum((cls_data['pred'] == 0) & (cls_data['true'] == 1))
    TN = np.sum((cls_data['pred'] == 0) & (cls_data['true'] == 0))
    print("TP {}".format(TP))
    print("FP {}".format(FP))
    print("FN {}".format(FN))
    print("TN {}".format(TN))
    print("Accuracy {:.2f}".format(metrics.accuracy_score(cls_data['true'], cls_data['pred'])))
    print("Precision {:.2f}".format(metrics.precision_score(cls_data['true'], cls_data['pred'])))
    print("Recall {:.2f}".format(metrics.recall_score(cls_data['true'], cls_data['pred'])))
    print("F {:.2f}".format(metrics.f1_score(cls_data['true'], cls_data['pred'])))


def scores():
    scores_df = pd.read_csv('scores.csv')
    max_call, name = 0, None
    for col in ['score_logreg', 'score_svm', 'score_knn' , 'score_tree']:
        score = metrics.roc_auc_score(scores_df['true'], scores_df[col])
        print("{} {:.2f}".format(col, score))
        precision, recall, thresholds = metrics.precision_recall_curve(scores_df['true'], scores_df[col])
        max_pre = np.max(precision[recall > 0.7])
        if max_pre > max_call:
            max_call = max_pre
            name = col
    print("BEST {} {:.2f}".format(name, max_call))


if __name__ == '__main__':
    # classification()
    scores()
