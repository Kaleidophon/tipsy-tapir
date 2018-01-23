from gensim.models import Word2Vec
import pyndri.compat
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


print("Training word embeddings...")


def build_sentences(pyndri_index):
    vocabulary = pyndri.extract_dictionary(pyndri_index)
    sentences = pyndri.compat.IndriSentences(pyndri_index, vocabulary)
    return sentences


def train_word_embeddings(sentences, epochs, epsilon=None, **model_parameters):
    word2vec = Word2Vec(**model_parameters)
    word2vec.build_vocab(sentences, trim_rule=None)

    last_loss = None
    losses = []
    model = word2vec
    print("Start training...")
    for epoch in range(epochs):
        print("Starting epoch #{}...".format(epoch+1))
        start = time.time()

        word2vec = model

        word2vec.train(sentences, total_examples=len(sentences), epochs=1, compute_loss=True)

        end = time.time()
        current_loss = word2vec.running_training_loss
        losses.append(current_loss)
        model = word2vec
        print("Epoch #{} took {:.2f} seconds.".format(epoch+1, end-start))
        print("Loss epoch #{}: {}".format(epoch+1, current_loss))

        if epsilon is not None and epoch > 0:
            # Check convergence criterion
            if np.abs(current_loss - last_loss) < epsilon:
                print("Training stopped after convergence criterion was reached in epoch #{}.".format(epoch+1))
                break

        last_loss = current_loss

    print("Training was stopped after #{} epochs.".format(epochs))
    return word2vec, losses


def save_word2vec_model(model, path):
    model.save(path)


def load_word2vec_model(path, to_train=False):
    model = Word2Vec.load(path)

    if to_train:
        return model

    # In case it doesn't need to be trained, delete train code to free up ram
    word_vectors = model.wv

    # TODO: Test this code
    context_vectors = dict()
    if hasattr(model, "syn1"):
        # For hierarchical softmax
        context_vectors = model.syn1
    elif hasattr(model, "syn1neg"):
        # For negative sampling
        context_vectors = model.syn1neg

    del model
    return word_vectors, context_vectors


WORD_EMBEDDING_PARAMS = {
    "alpha": 0.000025,  # Learning rate
    "size": 300,  # Embedding size
    "window": 5,  # One-sided window size
    "sg": True,  # Skip-gram.
    "min_count": 5,  # Minimum word frequency.
    "sample": 1e-3,  # Sub-sample threshold.
    "hs": False,  # Hierarchical softmax.
    "negative": 10,  # Number of negative examples.
    "iter": 1,  # Number of iterations.
    "workers": 4  # Number of workers
}


def plot_losses(losses, epochs):
    plt.plot(range(len(losses)), losses)
    plt.title("Training losses for Word2Vec model for {} iterations".format(epochs))
    plt.show()


if __name__ == "__main__":
    # Train
    EPOCHS = 20
    index = pyndri.Index('index/')
    sentences = build_sentences(index)
    #sentences_subset = [sentence for i, sentence in enumerate(sentences) if i < 25000]
    w2v_model, losses = train_word_embeddings(sentences, epochs=EPOCHS, **WORD_EMBEDDING_PARAMS)
    save_word2vec_model(w2v_model, "./w2v_test")
    plot_losses(losses, EPOCHS)

    # Load vectors
    #word_vectors, context_vectors = load_word2vec_model("./w2v_test")
    #pass
