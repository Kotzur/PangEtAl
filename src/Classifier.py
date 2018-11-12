from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src import utils
from src.Type import Type


class Classifier:
    def __init__(self, stemming=False, frequency=False, unigrams=True,
                 bigrams=False, class_type=Type.NB, feature_cut_off=0):
        self.stemming = stemming
        self.frequency = frequency
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.type = class_type
        self.feature_cut_off = feature_cut_off

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

        reviews = utils.load_reviews()
        if self.stemming:
            reviews = utils.stem(reviews)
        folds = utils.round_robin_split(reviews)

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
