import pandas as pd
import json
import numpy as np
import os
import time
import sentencepiece as spm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_breast_cancer

from acquisition_functions import RandomSelection, EntropySelection
from models import NBClassifier
# load data
toy_data = load_breast_cancer()
X = toy_data['data']
y = toy_data['target']

# SETTINGS for AL experiment:

# list with different settings for k (sample size):
Ks = [100]

# the number of times you want to repeat the experiment:
repeats = 1

# the warm start sample size:
start_sample_sizes = 25

# the different models to use (as defined in models.py):
models = [
          NBClassifier,
         ]

# the different acquisition/selection functions to use (as defined in acquisition_functions.py):
selection_functions = [
                        EntropySelection,
                        RandomSelection
                       ]

selection_functions_str = [
                        "EntropySelection",
                        "RandomSelection"
                       ]

trainset_size = len(X)
max_queried = trainset_size - Ks[-1]

d = {}
stopped_at = - 1

# # load MISP data
# csv_dir = '/content/drive/My Drive/UvA/master/Thesis/AL/MISP_train_Data.csv'
# MISP_data = pd.read_csv(csv_dir, encoding="utf-8")
#
# raw_X = list(MISP_data["sentence"])
# y = list(MISP_data["label"])
#
# # now add iACE data
# json_dir = "/content/drive/My Drive/UvA/master/Thesis/AL/sentence_labels_30-4.json"
# with open(json_dir) as f:
#     iACE_label_dict = json.load(f)
#
# json_dir = "/content/drive/My Drive/UvA/master/Thesis/AL/ioc_blog_sentences.json"
# with open(json_dir) as f:
#     iACE_sentence_dict = json.load(f)
#
# iACE_sentence_to_doc = {}
# for doc in iACE_sentence_dict:
#     for sentence in iACE_sentence_dict[doc]:
#         iACE_sentence_to_doc[sentence] = doc
#
# # create feature and label vectors
# for i in iACE_label_dict["IOC"]:
#     text = iACE_sentence_dict[iACE_sentence_to_doc[i]][i]
#     raw_X.append(text)
#     y.append(1)
# for i in iACE_label_dict["non_IOC"][:int(float(len(iACE_label_dict["IOC"]) * 2.3))]:
#     text = iACE_sentence_dict[iACE_sentence_to_doc[i]][i]
#     raw_X.append(text)
#     y.append(0)
