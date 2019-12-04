import numpy as np
import time
from sklearn import metrics
from sklearn.utils import check_random_state

def get_k_random_samples(random_seed, initial_labeled_samples, X_train_full,y_train_full):
    random_state = check_random_state(0)
    np.random.seed(random_seed)
    permutation = np.random.choice(X_train_full.shape[0],
                                   initial_labeled_samples,
                                   replace=False)
    print()
    print('initial random chosen samples', permutation.shape),
#            permutation)
    X_train = X_train_full[permutation]
    y_train = y_train_full[permutation]
    X_train = X_train.reshape((X_train.shape[0], -1))
    bin_count = np.bincount(y_train.astype('int32'))
    unique = np.unique(y_train.astype('int32'))
    print (
        'initial train set:',
        X_train.shape,
        y_train.shape,
        'unique(labels):',
        bin_count,
        unique,
        )
    return (permutation, X_train, y_train)


class TrainModel:

    def __init__(self, model_object):
        self.accuracies = []
        self.model_object = model_object()

    def print_model_type(self):
        print(self.model_object.model_type)

    # we train normally and get probabilities for the validation set.
    # i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, y_train, X_val, X_test, c_weight, extra_repr=None):
        print('Train set:', X_train.shape, 'y:', y_train.shape)
        print('Val   set:', X_val.shape)
        print('Test  set:', X_test.shape)
        t0 = time.time()

        if extra_repr:
            we_x_train, char_x_train = extra_repr[0], extra_repr[1]
            we_val, char_val = extra_repr[2], extra_repr[3]

            (X_train, X_val, X_test, self.val_y_predicted,
             self.test_y_predicted) = \
                self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight,
                                              we_x_train, char_x_train, we_val, char_val)

        else:
            (X_train, X_val, X_test, self.val_y_predicted,
             self.test_y_predicted) = \
                self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight)

        self.run_time = time.time() - t0
        return (
        X_train, X_val, X_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want accuracy only for the test set

    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        self.accuracies.append(classif_rate)
        print('--------------------------------')
        print('Iteration:', i)
        print('--------------------------------')
        print('y-test set:', y_test.shape)
        print('Example run in %.3f s' % self.run_time, '\n')
        print("Accuracy rate for %f " % (classif_rate))
        print("Classification report for classifier %s:\n%s\n" % (
        self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print('-------------------------------- \n\n')


class TheAlgorithm(object):
    accuracies = []

    def __init__(self, initial_labeled_samples, query_size, model_object, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.query_size = query_size
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, permutation, X_train_full, y_train_full, X_test, y_test, max_queried):
        # initialize process by applying base learner to labeled training data set to obtain Classifier
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        X_train = X_train_full[permutation]
        y_train = y_train_full[permutation]

        # assign the val set the rest of the 'unlabelled' training data
        X_val = np.array([])
        y_val = np.array([])
        X_val = np.copy(X_train_full)
        X_val = np.delete(X_val, permutation, axis=0)
        y_val = np.copy(y_train_full)
        y_val = np.delete(y_val, permutation, axis=0)


        # do the same for other data representations
        try:
            we_x_train = WE_X_train[permutation]
            char_x_train = CHAR_X_train[permutation]

            we_x_val = np.array([])
            we_x_val = np.copy(WE_X_train)
            we_x_val = np.delete(we_x_val, permutation, axis=0)

            char_x_val = np.array([])
            char_x_val = np.copy(CHAR_X_train)
            char_x_val = np.delete(char_x_val, permutation, axis=0)
        except:
            pass

        print('val set:', X_val.shape, y_val.shape, permutation.shape)
        print()

        active_iteration = 1

        # fit model
        self.clf_model = TrainModel(self.model_object)
        if self.model_object.__name__ in ["MultiChannelDNN", "BayesApproxDNN"]:
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced',
                                                            extra_repr=[we_x_train, char_x_train, we_x_val, char_x_val])
        else:
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')

        self.clf_model.get_test_accuracy(1, y_test)

        while self.queried < max_queried:
            active_iteration += 1

            # get validation probabilities
            if self.model_object.__name__ in ["BayesApproxDNN", "MLP", "MultiChannelDNN"]:
                if self.model_object.__name__ == "MLP":
                    probas_val = \
                        self.clf_model.model_object.classifier.predict(X_val)
                elif self.model_object.__name__ == "BayesApproxDNN":
                    probas = []
                    for t in range(10):
                        print("Forward:", t)
                        probas.append(self.clf_model.model_object.classifier.predict([we_x_val, char_x_val, X_val]))
                    probas_val = np.mean(probas, axis=0)

                else:
                    # print(we_x_val.shape, char_x_train.shape, X_val.shape)
                    probas_val = \
                        self.clf_model.model_object.classifier.predict([we_x_val, char_x_val, X_val])
                probas = []
                for prob in probas_val:
                    probas.append([float(1 - prob)])
                for index, prob in enumerate(probas_val):
                    probas[index].append(float(prob))
                probas_val = np.array(probas)

            else:
                probas_val = \
                    self.clf_model.model_object.classifier.predict_proba(X_val)

            print('val predicted:',
                  self.clf_model.val_y_predicted.shape,
                  self.clf_model.val_y_predicted)
            print('probabilities:', probas_val.shape, '\n',
                  np.argmax(probas_val, axis=1))

            # select samples using a selection function
            uncertain_samples = \
                self.sample_selection_function.select(probas_val, self.query_size)

            print('trainset before', X_train.shape, y_train.shape)
            X_train = np.concatenate((X_train, X_val[uncertain_samples]))
            y_train = np.concatenate((y_train, y_val[uncertain_samples]))
            print('trainset after', X_train.shape, y_train.shape)
            self.samplecount.append(X_train.shape[0])

            if self.model_object.__name__ in ["MultiChannelDNN", "BayesApproxDNN"]:
                # concat for other data representations
                we_x_train = np.concatenate((we_x_train, we_x_val[uncertain_samples]))
                char_x_train = np.concatenate((char_x_train, char_x_val[uncertain_samples]))

            bin_count = np.bincount(y_train.astype('int32'))
            unique = np.unique(y_train.astype('int32'))
            print(
                'updated train set:',
                X_train.shape,
                y_train.shape,
                'unique(labels):',
                bin_count,
                unique)

            X_val = np.delete(X_val, uncertain_samples, axis=0)
            y_val = np.delete(y_val, uncertain_samples, axis=0)

            if self.model_object.__name__ in ["MultiChannelDNN", "BayesApproxDNN"]:
                # delete for other data representations
                we_x_val = np.delete(we_x_val, uncertain_samples, axis=0)
                char_x_val = np.delete(char_x_val, uncertain_samples, axis=0)

            print('val set:', X_val.shape, y_val.shape)
            print()

            self.queried += self.query_size

            # (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')

            if self.model_object.__name__ in ["MultiChannelDNN", "BayesApproxDNN"]:
                (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced',
                                                                extra_repr=[we_x_train, char_x_train, we_x_val,
                                                                            char_x_val])
            else:
                (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')

            self.clf_model.get_test_accuracy(active_iteration, y_test)

        print('final active learning accuracies',
              self.clf_model.accuracies)


def experiment(X_train_full, y_train_full, X_test, y_test,
               d, models, selection_functions, Ks, init_sample_size, repeats, max_queried, contfrom):
    algos_temp = []
    count = 0

    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}

        for selection_function in selection_functions:
            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for k in Ks:
                d[model_object.__name__][selection_function.__name__][str(k)] = []

                for i in range(0, repeats):
                    count += 1
                    if count >= contfrom:
                        print('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (
                            count, model_object.__name__, selection_function.__name__, k, i))
                        alg = TheAlgorithm(init_sample_size,
                                           k,
                                           model_object,
                                           selection_function
                                           )
                        # create initial train and val set
                        (permutation, X_train, y_train) = get_k_random_samples(i, init_sample_size,
                                                                               X_train_full, y_train_full)
                        alg.run(permutation, X_train_full, y_train_full, X_test, y_test, max_queried)
                        d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
                        # fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                        # pickle_save(fname, d)
                        if count % 5 == 0:
                            print(json.dumps(d, indent=2, sort_keys=True))
                        print()
                        print('---------------------------- FINISHED ---------------------------')
                        print()
    return d