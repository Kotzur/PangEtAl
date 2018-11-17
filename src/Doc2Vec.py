import multiprocessing

import gensim
import glob
from random import shuffle
from collections import namedtuple, OrderedDict

from smart_open import smart_open

import src.utils as utils
import os
from gensim.test.utils import common_texts
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

imdb_dir = os.path.join(utils.DATA_DIR, "imdb")
clean_imdb_dir = os.path.join(utils.DATA_DIR, "clean_imdb")
folders = os.listdir(imdb_dir)
folders = [os.path.join(f, a) for f in folders for a in os.listdir(os.path.join(imdb_dir, f))]
all_data_filepath = os.path.join(clean_imdb_dir, "all_lines.txt")


def load_reviews():
    all_lines = []
    control_chars = [chr(0x85)]
    if not os.path.isfile(all_data_filepath):
        for folder in folders:
            output = folder.replace('/', '-') + '.txt'
            # Find all files in folder.
            txt_files = glob.glob(os.path.join(imdb_dir, folder, '*.txt'))
            print(" %s: %i files" % (folder, len(txt_files)))

            # Add normalized reviews of same category to one file.
            with smart_open(os.path.join(clean_imdb_dir, output), "wb") as combined_reviews:
                for index, txt_file in enumerate(txt_files):
                    with smart_open(txt_file, "rb") as file:
                        # Read the file.
                        review = file.read().decode("utf-8")
                        # Remove unknown chars.
                        for c in control_chars:
                            review = review.replace(c, ' ')
                        # Normalize data
                        review = utils.clean_data(review)
                        # Append to general list of lines.
                        all_lines.append(review)
                        # Write to file.
                        combined_reviews.write(review.encode("utf-8"))
                        combined_reviews.write("\n".encode("utf-8"))

            with smart_open(all_data_filepath, "wb") as output_file:
                for index, review in enumerate(all_lines):
                    line = f"{index} {review}\n"
                    output_file.write(line.encode("utf-8"))


SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')
load_reviews()

all_docs = []
with smart_open(all_data_filepath, 'rb', encoding='utf-8') as all_data:
    for line_no, line in enumerate(all_data):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no]
        # Choose the right split tag
        split = ['train', 'extra', 'extra', 'extra', 'extra', 'train', 'test', 'test'][line_no//12500]    # 12.5k train pos, 50k unsup,
        # Choose the right sentiment tag                                                # 12.5k train-neg, 25k test
        sentiment = [1.0, None, None, None, None, 0.0, 1.0, 0.0][line_no//12500]    # 12.5k pos, 50k unsup,
        # Add the doc tuple.                                                        # 12.5k neg, 12.5k pos, 12.5k neg
        all_docs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in all_docs if doc.split == 'train']
test_docs = [doc for doc in all_docs if doc.split == 'test']

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(all_docs), len(train_docs), len(test_docs)))

# Shuffle to remove concentrated data.
doc_list = all_docs[:]
shuffle(doc_list)

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1

simple_models = [
    # DBOW plain
    Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
    # DM with default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
    # DM with concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
]

for model in simple_models:
    model.build_vocab(all_docs)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)

models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])

