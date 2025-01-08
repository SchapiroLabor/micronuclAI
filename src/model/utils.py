import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle


def evaluate_binary_model(scores, labels):
    auc = roc_auc_score(labels, scores)

    scores_classified = [int(p >= 0.5) for p in scores]
    tn, fp, fn, tp = confusion_matrix(labels, scores_classified).ravel()
    balanced_accuracy = balanced_accuracy_score(labels, scores_classified)

    dict_metrics = {"auc": auc, "confusion_matrix": (tn, fp, fn, tp), "balanced_accuracy": balanced_accuracy}

    return dict_metrics


def plot_confusion_matrix(scores, labels):
    # Get confusion matrix
    cm = confusion_matrix(scores, labels)

    # Create canvas and plot h confusion amtrix
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax,)
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title("Confusion Matrix")
    sns.despine()

    return fig


def evaluate_multiclass_model(scores, labels):
    # Calculate MAE
    mae = mean_absolute_error(scores, labels)

    # Calculate MSE
    mse = mean_squared_error(scores, labels)

    # calculate RMSE
    rmse = np.sqrt(mse)

    # calculate Acuracy
    acc = accuracy_score(scores, labels)

    # calculate balanced accuracy
    acc_bal = balanced_accuracy_score(scores, labels)

    # Get F1
    f1_micro = f1_score(scores, labels, average="micro")
    f1_macro = f1_score(scores, labels, average="macro")
    f1_weighted = f1_score(scores, labels, average="weighted")
    
    # Micronuclei count
    n_micro = scores.sum() 

    # Cell count
    n_cells = scores.shape[0]

    # Micronuclei ratio
    mn_ratio = n_micro/n_cells

    # Micronuclei deviance
    mn_deviant = (labels.sum()-scores.sum())/labels.sum()

    # Gather results
    dict_metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "ACC": acc,
        "ACC_balanced": acc_bal,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "micronuclei": n_micro,
        "cells": n_cells,
        "micronuclei_cell_ratio": mn_ratio,
        "micronuclei_deviance": mn_deviant
    }

    # convert to dataframe
    metrics = pd.DataFrame(dict_metrics, index=range(1))

    return metrics


def create_lists_for_train_test_split(index_test_well, path_dict_well_images):
    index_test_well = index_test_well

    with open(path_dict_well_images, 'rb') as f:
        dict_well_images = pickle.load(f)

    list_wells_sorted = sorted(dict_well_images.keys())

    test_well = list_wells_sorted[index_test_well]
    train_wells = list_wells_sorted.copy()
    train_wells.remove(test_well)

    nested_list_train_images = [dict_well_images.get(key) for key in train_wells]
    list_train_images = [image for sublist in nested_list_train_images for image in sublist]

    list_test_images = dict_well_images.get(test_well)

    print("Training on " + str(len(list_train_images)) + " images.")
    print("Testing on " + str(len(list_test_images)) + " images.")

    return list_train_images, list_test_images, test_well
