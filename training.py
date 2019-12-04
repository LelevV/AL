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