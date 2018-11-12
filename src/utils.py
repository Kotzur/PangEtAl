import math
import os

from scipy.stats import binom

from tartarus.PorterStemmer import PorterStemmer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_FILES_PATH = os.path.join(ROOT_DIR, "..", "data", "neg_token")
POS_FILES_PATH = os.path.join(ROOT_DIR, "..", "data", "pos_token")


def sign_test(actual_classes, a_predictions, b_predictions, features):
    average_sum_p = 0
    for fold in range(0, len(actual_classes)):
        actual = actual_classes[fold]
        a_pred = a_predictions[fold]
        b_pred = b_predictions[fold]

        a_better = sum(1 for i, a in enumerate(actual) if
                       (a == a_pred[i]) and (not a == b_pred[i]))
        b_better = sum(1 for i, a in enumerate(actual) if
                       (a == b_pred[i]) and (not a == a_pred[i]))
        both_same = sum(1 for i, a in enumerate(actual) if
                         (a_pred[i] == b_pred[i]))

        print("NB better: %d" % a_better)
        print("SVM better: %d" % b_better)
        print("Both same: %d" % both_same)

        big_n = 2 * math.ceil(both_same / 2) + a_better + b_better
        k = math.ceil(both_same / 2) + min(a_better, b_better)
        p = 2 * sum(binom(big_n, i) * math.pow(0.5, i) * math.pow(0.5, big_n - i) for i in range(0, k))
        print("The probability for fold %d that the models are the same is %f" % (fold, p*100))
        average_sum_p += p

    average_p = average_sum_p / len(actual_classes)
    print("***")
    print("Average probability across folds that the models with %s are the same is %f" % (features, average_p*100))
    print("***")

    return average_p


def load_reviews():
    sentiments = [-1, 1]
    sentiment_file_paths = [NEG_FILES_PATH, POS_FILES_PATH]
    reviews = []
    for i, sentiment in enumerate(sentiments):
        sentiment_file_path = sentiment_file_paths[i]
        files = [f for f in os.listdir(sentiment_file_path) if os.path.isfile(os.path.join(sentiment_file_path, f))]
        for path in files:
            review = ""
            with open(os.path.join(sentiment_file_path, path)) as file:
                for word in file.read().splitlines():
                    review += " " + word
            file.close()
            reviews.append([review, sentiment])

    return reviews


def round_robin_split(reviews):
    ten_splits = []
    for i in range(10):
        ten_splits.append([reviews[r] for r in range(len(reviews)) if r % 10 == i])
    return ten_splits


def stem(reviews):
    porter_stemmer = PorterStemmer()
    stemmed_reviews = [[0, 0] for _ in range(len(reviews))]
    for i, review in enumerate(reviews):
        output = ''
        for token in review[0].split():
            if token.isalpha():
                output += porter_stemmer.stem(token.lower(), 0, len(token) - 1)
            else:
                output += token
            output += " "
        stemmed_reviews[i][0] = output
        stemmed_reviews[i][1] = review[1]
    return stemmed_reviews


def run_experiment(nb_class, svm_class, text):
    nb_act, nb_pred = nb_class.classify()
    svm_act, svm_pred = svm_class.classify()
    sign_test(nb_act, nb_pred, svm_pred, text)
    nb_class.eval_and_print(nb_act, nb_pred)
    svm_class.eval_and_print(svm_act, svm_pred)