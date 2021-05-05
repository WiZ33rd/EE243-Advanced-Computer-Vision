import numpy as np
import matplotlib.pyplot as plt


def plot_trainaccuracy():
    train_accuracy = np.load('train_accuracy.npy')
    plt.plot(train_accuracy)
    plt.xlabel('Training iterations')
    plt.ylabel('Train accuracy')
    plt.show()


def plot_testaccuracy():
    test_accuracy = np.load('test_accuracy.npy')
    plt.plot(test_accuracy)
    plt.xlabel('Training iterations')
    plt.ylabel('Test accuracy')
    plt.show()


def plot_loss():
    train_loss = np.load('train_loss.npy')
    plt.plot(train_loss)
    plt.xlabel('Training iterations')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    plot_trainaccuracy()
    plot_testaccuracy()
    plot_loss()
