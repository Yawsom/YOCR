import numpy as np

DATA_DIR = "MNIST_data/"

TRAIN_DATA_FILENAME = DATA_DIR + "train-images.idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "train-labels.idx1-ubyte"

TEST_DATA_FILENAME = DATA_DIR + "t10k-images.idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "t10k-labels.idx1-ubyte"

class MLP_layer:
    def __init__(self, input_size, output_size):

        limit = np.sqrt(6 / (input_size + output_size))
        #uniform xiaver initialization
        self.Weights = np.random.uniform(low=-limit,high=limit,size=(output_size, input_size))
        self.biases = np.zeros(output_size)
        self.zCache = None
        self.aCache = None
        self.xCache = None
        self.learning_rate = 0.01

    # activation function is sigmoid

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.xCache = x
        self.zCache = np.dot(self.Weights, x) + self.biases
        self.aCache = self.activation_function(self.zCache)
        return self.aCache 

    def backward(self, gradient_input):

        # gradient_input is dL/da for this layer's output.
        # Multiply by da/dz to obtain this layer's delta = dL/dz.
        # da/dz = a * (1 - a)

        delta = gradient_input * (self.aCache * (1 - self.aCache))
        
        delta_weights = np.outer(delta, self.xCache)
        delta_biases = delta

        grad_input = self.Weights.T @ delta
        self.Weights -= self.learning_rate * delta_weights
        self.biases -= self.learning_rate * delta_biases

        return grad_input

class MLP_layer_output(MLP_layer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        
    def backward(self, y_true):

        delta = ((self.aCache - y_true) * self.aCache * (1 - self.aCache))
        
        delta_weights = np.outer(delta, self.xCache)
        delta_biases = delta
        grad_input = self.Weights.T @ delta
        self.Weights -= self.learning_rate * delta_weights
        self.biases -= self.learning_rate * delta_biases

        return grad_input
    

class MLP:

    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        for i in range(len(hidden_sizes)):
            self.layers.append(MLP_layer(input_size, hidden_sizes[i]))
            input_size = hidden_sizes[i]
        self.layers.append(MLP_layer_output(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true):
        grad_input = self.layers[-1].backward(y_true)
        for layer in reversed(self.layers[:-1]):
            grad_input = layer.backward(grad_input)
        return grad_input

    def classify(self, x):
        return np.argmax(self.forward(x))

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
    flattened = [flatten_list(sample) for sample in X]
    return [np.array(sample, dtype=np.float64) / 255.0 for sample in flattened]

def flatten_list(list: list) -> list:
    return [item for sublist in list for item in sublist]
    
def accuracy_score(y_true, y_pred) -> float:
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def main():
    X_train = extract_features(read_images(TRAIN_DATA_FILENAME))
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = extract_features(read_images(TEST_DATA_FILENAME))
    y_test = read_labels(TEST_LABELS_FILENAME)

    # TODO: implement MLP (squared-error loss)
    mlp = MLP(784, [100, 100], 10)

    for epoch in range(10):
        order = np.random.permutation(len(X_train))
        for i in order:
            x = X_train[i]
            y = y_train[i]
            y = one_hot_encode(y, 10)
            mlp.forward(x)
            mlp.backward(y)

    y_pred = [mlp.classify(x) for x in X_test]
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()