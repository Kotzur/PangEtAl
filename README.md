# Replication of Pang et Al #
This mini-project replicates work done on sentiment analysis using NB and SVM by Pang et al. Coursework for NLP Units of Assessment.

## File structure ##
After unpacking the zip, the following filestructure is found:

- Data: folders with positive and negative reviews, raw and tokenized, indicated by dir name.
- Src: main folder with source code.
- Tartarus: source code for Porter stemming algorithm used. I adapted its main function call into function stem() in 
exercise1.py.

## Dependencies ##
The project is based primarily on `sklearn` and has the following dependencies:

- `enum`
- `scipy`
- `sklearn`:
    - `metrics`
    - `naive_bayes`
    - `feature_extraction.text`
    - `pipeline`
    - `svm`
- `Tartarus` Porter Stemmer (available in the zip)

## Workflow ##
The whole exercise 1 is placed in a single file `exercise1.py`. It contains a class `Classifier` which is used to represent 
both NB and SVM depending on constructor arguments.

### Defining classifiers ###
To create a classifier, provide its type in form of an enum.
Any of the following boolean arguements can be passed to the constructor to adjust the models:

- stemming
- frequency (presence otherwise)
- unigrams
- bigrams
- feature_cutoff

By default, classifiers are NB no stemming, presence, unigrams with feature cut off 0.

### Running experiments ###
To run an experiment, call the run_experiment method, passing nb and svm models to it together with the text describing 
what features were set to true in the models (to help with clearer print).

If the file is just run, all experiments for feaature cut off = 0 that are reported in Report 1 will be ran.