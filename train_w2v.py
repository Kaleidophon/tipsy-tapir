import copy
import gensim
import logging
import pyndri
import pyndri.compat
import numpy as np
import matplotlib.pyplot as plt
import time


def train_word_embeddings(sentences, epochs, epsilon=None, save_path=None, **model_parameters):
    """ Train word embeddings using Gensim's Word2Vec implementation. """
    logging.info('Constructing word2vec vocabulary.')
    word2vec_init = gensim.models.Word2Vec(**model_parameters)

    # Build vocab.
    word2vec_init.build_vocab(sentences, trim_rule=None)

    models = [word2vec_init]
    last_loss = None
    losses = []
    for epoch in range(1, epochs + 1):
        logging.info('Epoch %d', epoch)
        start = time.time()

        model = copy.deepcopy(models.pop())
        model.train(sentences, compute_loss=True, epochs=model.iter, total_examples=len(sentences))

        end = time.time()
        current_loss = model.get_latest_training_loss()
        losses.append(current_loss)

        logging.info("Epoch #{} took {:.2f} seconds.".format(epoch, end - start))
        logging.info("Loss epoch #{}: {}".format(epoch, current_loss))
        models.append(model)

        if epsilon is not None and epoch > 1:
            # Check convergence criterion
            if np.abs(current_loss - last_loss) < epsilon:
                logging.info("Training stopped after convergence criterion was reached in epoch #{}.".format(epoch))
                break

        if save_path is not None:
            save_word2vec_model(models[-1], save_path)

        last_loss = current_loss

    logging.info('Trained models: %s', models)
    logging.info("Training was stopped after #{} epochs.".format(epochs))
    return models, losses


def save_word2vec_model(model, path):
    """ Save the model to a path. """
    model.save(path)


def build_sentences(pyndri_index):
    """ Build sentences from the Pyndri Index. """
    dictionary = pyndri.extract_dictionary(pyndri_index)
    sentences = pyndri.compat.IndriSentences(pyndri_index, dictionary)
    return sentences


def plot_losses(losses, epochs):
    """ Create a nice plot of the loss values during the training epochs. """
    plt.plot(range(len(losses)), losses)
    plt.title("Training losses for Word2Vec model for {} iterations".format(epochs))
    plt.show()


if __name__ == "__main__":
    EPOCHS = 60
    WORD_EMBEDDING_PARAMS = {
        "size": 300,  # Embedding size
        "window": 5,  # One-sided window size
        "sg": True,  # Skip-gram.
        "min_count": 5,  # Minimum word frequency.
        "sample": 1e-3,  # Sub-sample threshold.
        "hs": False,  # Hierarchical softmax.
        "negative": 10,  # Number of negative examples.
        "iter": 1,  # Number of iterations.
        "workers": 8,  # Number of workers.
    }

    logging.basicConfig(level=logging.INFO)
    logging.info('Initializing word2vec.')

    index = pyndri.Index('index/')
    logging.info('Loading vocabulary.')
    sentences = build_sentences(index)

    w2v_models, losses = train_word_embeddings(
        sentences, epochs=EPOCHS, epsilon=100, save_path="./w2v_60", **WORD_EMBEDDING_PARAMS
    )
    plot_losses(losses, EPOCHS)
