from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from acquisition_functions import RandomSelection, EntropySelection
from models import NBClassifier
from al_cycle import experiment

# load data
toy_data = load_breast_cancer()
X = toy_data['data']
y = toy_data['target']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, shuffle=True, test_size=0.1, random_state=123)


# SETTINGS for AL experiment:

# list with different settings for k (sample size):
Ks = [10]

# the number of times you want to repeat the experiment:
repeats = 1

# the warm start sample size:
start_sample_sizes = 10

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

trainset_size = len(X_train_full)
max_queried = trainset_size - Ks[-1]

d = {} # empty dict to put the results in
stopped_at = - 1 # parameter which can be used to complete an unfinished experiment

# run the experiment
d = experiment(X_train_full,
               y_train_full,
               X_test,
               y_test,
               d,
               models,
               selection_functions,
               Ks,
               start_sample_sizes,
               repeats,
               max_queried,
               stopped_at + 1)

print(d)