class RandomSelection(object):

    def select(probas_val, query_size):
        random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], query_size, replace=False)

        #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

        return selection

class EntropySelection(object):

    def select(probas_val, query_size):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:query_size]
        return selection
