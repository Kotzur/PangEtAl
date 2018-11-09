import math
import os

from enum import Enum

from scipy.special import binom
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tartarus.PorterStemmer import PorterStemmer


class Type(Enum):
    NB = 1
    SVM = 2


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NEG_FILES_PATH = os.path.join(ROOT_DIR, "..", "data", "neg_token")
POS_FILES_PATH = os.path.join(ROOT_DIR, "..", "data", "pos_token")
PICKLE_PATH = os.path.join(ROOT_DIR, "..", "data", "pickles")


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


class Classifier:
        def __init__(self, stemming=False, frequency=False, unigrams=True, bigrams=False, type=Type.NB, feature_cut_off=0):
            self.stemming = stemming
            self.frequency = frequency
            self.unigrams = unigrams
            self.bigrams = bigrams
            self.type = type
            self.feature_cut_off = feature_cut_off

        def load_reviews(self):
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

        def round_robin_split(self, reviews):
            ten_splits = []
            for i in range(10):
                ten_splits.append([reviews[r] for r in range(len(reviews)) if r % 10 == i])
            return ten_splits

        def stem(self, reviews):
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

        def eval_and_print(self, actual, predictions):
            total_accuracy = sum(
                metrics.accuracy_score(actual[i], predictions[i], normalize=True) for i in range(0, len(actual)))

            stemming_string = " with stemming" if self.stemming else ""

            frequency_string = "counting frequency" if self.frequency else "counting occurences"

            ngrams_string = "with unigrams"
            if not self.unigrams:
                ngrams_string = "with bigrams"
            elif self.bigrams:
                ngrams_string = "with unigrams and bigrams"

            classifier_string = "SVM" if self.type == Type.SVM else "NB"

            print("%s %s %s 10-fold accuracy%s: %f%%" % (classifier_string,
                                                         frequency_string,
                                                         ngrams_string,
                                                         stemming_string,
                                                         total_accuracy * 10))

        def train_classifier(self, train):
            min_ngram = 1 if self.unigrams else 2
            max_ngram = 2 if self.bigrams else 1
            feature_extractor = CountVectorizer(binary=False,
                                                ngram_range=(min_ngram, max_ngram),
                                                min_df=self.feature_cut_off) \
                if self.frequency else CountVectorizer(binary=True,
                                                       ngram_range=(min_ngram, max_ngram),
                                                       min_df=self.feature_cut_off)

            clf = SVC(gamma='auto', kernel='linear') if self.type == Type.SVM else MultinomialNB(alpha=1)
            classifier = Pipeline([('vect', feature_extractor),
                                   ('clf', clf)])
            classifier.fit([t[0] for t in train], [t[1] for t in train])
            return classifier

        def classify(self):
            if not (self.unigrams or self.bigrams):
                print("At least one ngram option must be chosen. Unigrams will be used as default.")
                unigrams = True

            reviews = self.load_reviews()
            if self.stemming:
                reviews = self.stem(reviews)
            folds = self.round_robin_split(reviews)

            predictions = []
            actual = []
            feature_count = 0
            print("folding")
            for fold_ind in range(len(folds)):
                print(fold_ind)
                train_lists = folds[:fold_ind] + folds[fold_ind + 1:]
                train = [item for sublist in train_lists for item in sublist]

                classifier = self.train_classifier(train)
                feature_count += len(classifier.named_steps['vect'].get_feature_names())

                test = folds[fold_ind]
                predictions.append(classifier.predict([t[0] for t in test]))
                actual.append([t[1] for t in test])

            feature_count = feature_count / 10
            print("Number of features on average: %f" % feature_count)
            return actual, predictions


if __name__ == '__main__':
    nb_classifier = Classifier(type=Type.NB, feature_cut_off=0)
    svm_classifier = Classifier(type=Type.SVM, feature_cut_off=0)

    # print("unigrams")
    # # unigrams
    # nb_act, nb_pred = nb_classifier.classify()
    # svm_act, svm_pred = svm_classifier.classify()
    # sign_test(nb_act, nb_pred, svm_pred, "unigrams")
    # nb_classifier.eval_and_print(nb_act, nb_pred)
    # svm_classifier.eval_and_print(svm_act, svm_pred)
    #
    # # unigrams + stemming
    # nb_classifier.stemming = True
    # svm_classifier.stemming = True
    # nb_act, nb_pred = nb_classifier.classify()
    # svm_act, svm_pred = svm_classifier.classify()
    # sign_test(nb_act, nb_pred, svm_pred, "unigrams and stemming")
    # nb_classifier.eval_and_print(nb_act, nb_pred)
    # svm_classifier.eval_and_print(svm_act, svm_pred)
    #
    # # unigrams + frequency
    # nb_classifier.frequency = True
    # svm_classifier.frequency = True
    # nb_act, nb_pred = nb_classifier.classify()
    # svm_act, svm_pred = svm_classifier.classify()
    # sign_test(nb_act, nb_pred, svm_pred, "unigrams")
    # nb_classifier.eval_and_print(nb_act, nb_pred)
    # svm_classifier.eval_and_print(svm_act, svm_pred)
    #
    # # unigrams + stemming + frequency
    # nb_classifier.stemming = False
    # svm_classifier.stemming = False
    # nb_act, nb_pred = nb_classifier.classify()
    # svm_act, svm_pred = svm_classifier.classify()
    # sign_test(nb_act, nb_pred, svm_pred, "unigrams and stemming")
    # nb_classifier.eval_and_print(nb_act, nb_pred)
    # svm_classifier.eval_and_print(svm_act, svm_pred)

    print("bigrams")
    # bigrams + stemming
    nb_classifier.stemming=True
    svm_classifier.stemming=True
    nb_classifier.frequency=False
    svm_classifier.frequency=False
    nb_classifier.unigrams=False
    nb_classifier.bigrams=True
    svm_classifier.unigrams=False
    svm_classifier.bigrams=True

    nb_act, nb_pred = nb_classifier.classify()
    svm_act, svm_pred = svm_classifier.classify()
    sign_test(nb_act, nb_pred, svm_pred, "bigrams and stemming")
    nb_classifier.eval_and_print(nb_act, nb_pred)
    svm_classifier.eval_and_print(svm_act, svm_pred)

    # bigrams
    nb_classifier.stemming = False
    svm_classifier.stemming = False
    nb_act, nb_pred = nb_classifier.classify()
    svm_act, svm_pred = svm_classifier.classify()
    sign_test(nb_act, nb_pred, svm_pred, "bigrams")
    nb_classifier.eval_and_print(nb_act, nb_pred)
    svm_classifier.eval_and_print(svm_act, svm_pred)

    print("unigrams and bigrams")
    # unigrams + bigrams
    nb_classifier.unigrams = True
    svm_classifier.unigrams = True
    nb_act, nb_pred = nb_classifier.classify()
    svm_act, svm_pred = svm_classifier.classify()
    sign_test(nb_act, nb_pred, svm_pred, "unigrams and bigrams")
    nb_classifier.eval_and_print(nb_act, nb_pred)
    svm_classifier.eval_and_print(svm_act, svm_pred)

    # unigrams + bigrams + stemming
    nb_classifier.stemming = True
    svm_classifier.stemming = True
    nb_act, nb_pred = nb_classifier.classify()
    svm_act, svm_pred = svm_classifier.classify()
    sign_test(nb_act, nb_pred, svm_pred, "unigrams and bigrams and stemming")
    nb_classifier.eval_and_print(nb_act, nb_pred)
    svm_classifier.eval_and_print(svm_act, svm_pred)
