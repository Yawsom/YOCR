import math
import heapq

DATA_DIR = "MNIST_data/"

TRAIN_DATA_FILENAME = DATA_DIR + "train-images.idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "train-labels.idx1-ubyte"

TEST_DATA_FILENAME = DATA_DIR + "t10k-images.idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "t10k-labels.idx1-ubyte"



def read_images(filename, nmax_images = None) -> list[list[list[int]]]:
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


def read_labels(filename, nmax_labels = None) -> list[int]:
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



def knn(X_train, y_train, X_test, k = 3, n_labels = 10):

    y_pred = []
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    # Calculate distances matrix 

    for sample in X_test:

        
        distances = []

        for train_sample in X_train:
            distance = euclidean_distance(sample,train_sample)
            distances.append(distance)
        
        #kth nearest terms
        k_closest_items = k_closest(distances, y_train, k)
        k_dict = {}

        for item in k_closest_items:
            label = item[1]

            if(label in k_dict):
              k_dict[label][1] += item[0]
              k_dict[label][2] += 1

            else:
                k_dict[label] = [label, item[0], 1]
        
        closest_label = None 
        max_count = 0
        for label, count in k_dict.items():
            if count[2] > max_count:
                max_count = count[2]
                closest_label = count 

            elif count[2] == max_count:
                if count[1] < closest_label[1]:
                    closest_label = count

        y_pred.append(closest_label[0])

    return y_pred



def k_closest(distances, y_train,  k) -> list[tuple]:
        min_heap = []

        for i in range(len(distances)):
            heapq.heappush(min_heap, (distances[i], y_train[i]))

        k_closest_items = []

        for i in range(k):
            k_closest_items.append(heapq.heappop(min_heap))

        return k_closest_items




def euclidean_distance(l1 : list[int], l2: [list[int]]) -> float:
    sum = 0

    if(len(l1) != len(l2)):
        raise ValueError("List length mismatch.")

    for item1, item2 in zip(l1, l2):
        sum += (item1-item2)*(item1-item2)


    return math.sqrt(sum)


def accuracy_score(y_true, y_pred) -> float:
    return sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)


def main():
    X_train = read_images(TRAIN_DATA_FILENAME)
    y_train = read_labels(TRAIN_LABELS_FILENAME)
    X_test = read_images(TEST_DATA_FILENAME)
    y_test = read_labels(TEST_LABELS_FILENAME)

    y_pred = knn(X_train, y_train, X_test, k = 5, n_labels = 10)
    accuracy = accuracy_score(y_test, y_pred)

    print(accuracy)





if __name__ == "__main__":
    main()