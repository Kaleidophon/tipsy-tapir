import copy
import pickle
import time

# EXT
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch import functional
import numpy as np
from matplotlib import pyplot as plt
import torch.utils.data as data


class LinearRanker(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        x_var = Variable(x)
        out = self.linear(x_var)
        return out


def train(model, dataset, loss=nn.MSELoss(), learning_rate=0.00001, iterations=10, batch_size=50,
          print_losses=True, log_file="./training.torch"):
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    train_loader = data.DataLoader(dataset, batch_size=batch_size)
    batch_print = pick_batch_print(dataset)
    losses = []

    train_start = time.perf_counter()

    previous_loss = np.inf
    for i in range(iterations):
        start = time.perf_counter()
        train_err = 0

        for j, (inputs, outputs) in enumerate(train_loader):
            start_batch = time.time()

            model.zero_grad()
            err = loss(model(inputs), outputs)
            train_err += err

            # Calculate gradients, adapt parameters
            err.backward()
            optimizer.step()

            end_batch = time.time()

            if (j + 1) % batch_print == 0:
                print(
                    "\rBatch #{}/{} finished in {:.4f} seconds.".format(i+1, j+1, end_batch - start_batch),
                    flush=True, end=""
                )

        end = time.perf_counter()
        print("Iteration #{} took {:.2f} seconds.".format(i+1, end - start))
        print("Training error iteration #{}: {}".format(i+1, train_err.double().data.numpy()[0]))
        print()
        losses.append(train_err.double().data.numpy()[0])

        if np.isclose(train_err.data.numpy(), previous_loss, atol=1e-6, rtol=1e-6): break
        previous_loss = train_err.data.numpy()

    save_torch_model(model, log_file)

    train_end = time.perf_counter()
    train_time = train_end - train_start
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)
    print("Total training time was {} hour(s), {} minute(s) and {:.2f} seconds".format(h, m, s))

    if print_losses:
        print(losses)
        plt.title(
            "Training loss of {} on\n{} over {} iterations".format(
                type(model).__name__, dataset.name, iterations
            ), size=12
        )
        plt.plot(range(len(losses)), losses)
        plt.show()

    return losses, train_time


def pick_batch_print(dataset):
    size = len(dataset)
    if size < 100:
        return 1
    elif size < 1000:
        return 20
    elif size < 10000:
        return 50
    elif size < 100000:
        return 100


def save_torch_model(model, path, whole_model=False):
    if whole_model:
        copied_model = copy.deepcopy(model)
        with open(path, "wb") as output_file:
            pickle.dump(copied_model, output_file)
    else:
        torch.save(model.state_dict(), path)


def load_torch_model(model_cls, path, **model_params):
    model = model_cls(**model_params)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    pass