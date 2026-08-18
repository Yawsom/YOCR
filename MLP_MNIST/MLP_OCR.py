import numpy as np

DATA_DIR = "KNN_MNIST/data/"

TRAIN_DATA_FILENAME = DATA_DIR + "train-images.idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "train-labels.idx1-ubyte"

TEST_DATA_FILENAME = DATA_DIR + "t10k-images.idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "t10k-labels.idx1-ubyte"

class MLP_layer:
    def __init__(self, input_size, output_size):
        self.Weights = np.random.rand(input_size, output_size)
        self.biases = np.zeros(output_size)

    def forward(self, x):
        return np.dot(x, self.Weights) + self.biases
        
    

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        for i in range(len(hidden_sizes)):
            self.layers.append(MLP_layer(input_size, hidden_sizes[i]))
            input_size = hidden_sizes[i]
        self.layers.append(MLP_layer(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            layer.forward(x)

    def backward(self):
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].backward(self.layers[i-1])



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


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def flatten_list(list: list) -> list:
    return [item for sublist in list for item in sublist]


def accuracy_score(y_true, y_pred) -> float:
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)


def main():
    X_train = read_images(TRAIN_DATA_FILENAME)
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME)
    y_test = read_labels(TEST_LABELS_FILENAME)

    # TODO: implement MLP (squared-error loss)


if __name__ == "__main__":
    main()
