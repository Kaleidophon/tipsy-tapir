from sklearn.linear_model import LogisticRegression

def train(dataset):
    model = LogisticRegression(n_jobs=4, max_iter=100)
    model.fit(dataset.inputs, dataset.outputs)
    return model
