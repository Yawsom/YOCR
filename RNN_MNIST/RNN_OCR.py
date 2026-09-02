import numpy as np

DATA_DIR = "MNIST_data/"

TRAIN_DATA_FILENAME = DATA_DIR + "train-images.idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "train-labels.idx1-ubyte"

TEST_DATA_FILENAME = DATA_DIR + "t10k-images.idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "t10k-labels.idx1-ubyte"

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.01
        # TODO: initialize W_xh, W_hh, b_h, W_hy, b_y

    def forward(self, x):
        # x is a sequence of shape (T, input_size) — T=28 columns for MNIST
        # TODO: unroll h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
        raise NotImplementedError

    def backward(self, y_true):
        # TODO: backprop through time
        raise NotImplementedError

    def classify(self, x):
        # TODO: last hidden state -> 10-way prediction
        raise NotImplementedError


def read_images(filename, nmax_images=None) -> list[list[list[int]]]:
    with open(filename, "rb") as fp:
        magic_number = int.from_bytes(fp.read(4), "big")
        if magic_number != 2051:
            raise ValueError("Invalid magic number in file: " + filename)

        num_images = int.from_bytes(fp.read(4), "big")
        num_rows = int.from_bytes(fp.read(4), "big")
        num_cols = int.from_bytes(fp.read(4), "big")
        images = []

        if nmax_images is not None:
            num_images = min(num_images, nmax_images)

        for _ in range(num_images):
            image = []
            for _ in range(num_rows):
                image.append(list(fp.read(num_cols)))
            images.append(image)

    return images


def read_labels(filename, nmax_labels=None) -> list[int]:
    with open(filename, "rb") as fp:
        magic_number = int.from_bytes(fp.read(4), "big")
        if magic_number != 2049:
            raise ValueError("Invalid magic number in file: " + filename)
        num_labels = int.from_bytes(fp.read(4), "big")
        labels = []

        if nmax_labels is not None:
            num_labels = min(num_labels, nmax_labels)

        for _ in range(num_labels):
            labels.append(int.from_bytes(fp.read(1), "big"))

    return labels


def accuracy_score(y_true, y_pred) -> float:
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


def extract_sequences(X):
    # each image -> (28, 28): timestep t is column t, pixels top to bottom
    sequences = []
    for sample in X:
        image = np.array(sample, dtype=np.float64) / 255.0
        sequences.append(image.T)
    return sequences

def extract_sequences_from_images(X):
    # each image -> (28, 28): timestep t is column t, pixels top to bottom
    sequences = []
    for sample in X:
        image = np.array(sample, dtype=np.float64) / 255.0
        sequences.append(image.T)
    return sequences    

def main():
    X_train = extract_sequences(read_images(TRAIN_DATA_FILENAME))
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = extract_sequences(read_images(TEST_DATA_FILENAME))
    y_test = read_labels(TEST_LABELS_FILENAME)

    # TODO: implement RNN (BPTT)
    rnn = RNN(28, 64, 10)

    for epoch in range(10):
        order = np.random.permutation(len(X_train))
        for i in order:
            x = X_train[i]
            y = y_train[i]
            y = one_hot_encode(y, 10)
            rnn.forward(x)
            rnn.backward(y)

    y_pred = [rnn.classify(x) for x in X_test]
    print(accuracy_score(y_test, y_pred))



if __name__ == "__main__":
    main()
