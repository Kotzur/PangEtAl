from src.Type import Type
from src.Classifier import Classifier
from src.utils import run_experiment

if __name__ == '__main__':
    nb_classifier = Classifier(class_type=Type.NB, feature_cut_off=0)
    svm_classifier = Classifier(class_type=Type.SVM, feature_cut_off=0)

    print("unigrams")
    # unigrams
    run_experiment(nb_classifier, svm_classifier, "unigrams")

    # unigrams + stemming
    nb_classifier.stemming = True
    svm_classifier.stemming = True
    run_experiment(nb_classifier, svm_classifier, "unigrams and stemming")

    # unigrams + frequency
    nb_classifier.frequency = True
    svm_classifier.frequency = True
    run_experiment(nb_classifier, svm_classifier, "unigrams, stemming and frequency")

    # unigrams + stemming + frequency
    nb_classifier.stemming = False
    svm_classifier.stemming = False
    run_experiment(nb_classifier, svm_classifier, "unigrams and frequency")

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
    run_experiment(nb_classifier, svm_classifier, "bigrams and stemming")

    # bigrams
    nb_classifier.stemming = False
    svm_classifier.stemming = False
    run_experiment(nb_classifier, svm_classifier, "bigrams")

    print("unigrams and bigrams")
    # unigrams + bigrams
    nb_classifier.unigrams = True
    svm_classifier.unigrams = True
    run_experiment(nb_classifier, svm_classifier, "unigrams and bigrams")

    # unigrams + bigrams + stemming
    nb_classifier.stemming = True
    svm_classifier.stemming = True
    run_experiment(nb_classifier, svm_classifier, "unigrams and bigrams and stemming")
