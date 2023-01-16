from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
import pickle


def evaluate_binary_model(scores, labels):
    auc = roc_auc_score(labels, scores)

    scores_classified = [int(p >= 0.5) for p in scores]
    tn, fp, fn, tp = confusion_matrix(labels, scores_classified).ravel()
    balanced_accuracy = balanced_accuracy_score(labels, scores_classified)

    dict_metrics = {"auc": auc, "confusion_matrix": (tn, fp, fn, tp), "balanced_accuracy": balanced_accuracy}

    return dict_metrics


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