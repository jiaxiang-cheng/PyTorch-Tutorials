""" source: https://nextjournal.com/gkoehler/pytorch-mnist """

import torchvision
import torch.optim as optim

from training import *
from testing import *
from network import *
from visualize import *

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/tmp/mnist/data',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/tmp/mnist/data',
                                   train=False,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = load_mnist()

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    plot_examples_before(example_data, example_targets)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses, train_counter, test_losses = [], [], []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(model, test_loader, test_losses)

    for epoch in range(1, n_epochs + 1):
        train(epoch, model, optimizer, train_loader, train_losses, train_counter)
        test(model, test_loader, test_losses)

    plot_training_curve(train_counter, train_losses, test_counter, test_losses)

    with torch.no_grad():
        output = model(example_data)

    plot_examples_after(example_data, output)

    # =================================== Continued Training from Checkpoints ======================================== #

    # continued_network = Net()
    # continued_optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # network_state_dict = torch.load()
    # continued_network.load_state_dict(network_state_dict)

    # optimizer_state_dict = torch.load()
    # continued_optimizer.load_state_dict(optimizer_state_dict)

    # for i in range(4, 9):
    #     test_counter.append(i * len(train_loader.dataset))
    #     train(i)
    #     test()

    # fig = plt.figure()
    # plt.plot(train_counter, train_losses, color='blue')
    # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()
