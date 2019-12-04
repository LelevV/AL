class NBClassifier(object):

    model_type = 'Naive Bayes Classifier'
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print ('training Naive Bayes Classifier...')
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)