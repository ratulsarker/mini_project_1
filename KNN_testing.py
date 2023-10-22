import numpy as np
import os


class KNN:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

def evaluate_acc(y_true, y_pred):
    accuracy = np.mean(y_true == y_pred)
    return accuracy


def k_fold_cross_validation(model, X, y, k):
    fold_size = len(X) // k
    accuracies = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size

        X_test = X[start:end]
        y_test = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = evaluate_acc(y_test, y_pred)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

#########################################################################

script_directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_directory, 'adult_num.npy')

adult_data = np.load(filename, allow_pickle=True, fix_imports=True)  # Load data from .npy file


missing_rows = [i for i in range(adult_data.shape[0]) if '?' in adult_data[i]]
clean_adult_data = np.delete(adult_data, missing_rows, axis=0) # Delete rows with missing data
column_names = clean_adult_data[0]

# Columns indices to be one-hot encoded
columns_to_encode = [1, 3, 5, 6, 7, 8, 9, 13]

# Prepare the new dataset with the original columns not encoded
adult_x = np.empty((clean_adult_data.shape[0] - 1, clean_adult_data.shape[1] - 1))
attributes = clean_adult_data[1:, :-1]

    # Function to encode a column and get new column names

for i , column in enumerate(attributes.T):
    scale_column = np.zeros(len(column))
    if i in columns_to_encode:

        categories = np.unique(column)

        categories_dict = {}

        for category in categories:
            categories_dict[category] = len(categories_dict)

        for j, val in enumerate(column):
            scale_column[j] = int(categories_dict[val])
        
    else:
        for j in range(len(scale_column)):
            scale_column[j] = column[j]

    # Mean feature scaling all features                   
    min_val = np.min(scale_column)
    max_val = np.max(scale_column)

    new_column = (scale_column - min_val)/(max_val - min_val)

    adult_x[:,i] = new_column

adult_y = np.zeros((clean_adult_data.shape[0] - 1, 1))

for i, val, in enumerate(clean_adult_data[1:, -1]):
    if '>' in val:
        adult_y[i] = 0

    else:
        adult_y[i] = 1


#######################################  car  #######################################
filename = os.path.join(script_directory, 'car_num.npy')

car_data = np.load(filename, allow_pickle=True, fix_imports=True)  # Load data from .npy file

missing_rows = [i for i in range(car_data.shape[0]) if '?' in car_data[i]]
clean_car_data = np.delete(car_data, missing_rows, axis=0) # Delete rows with missing data
column_names = clean_car_data[0]

# Prepare the new dataset with the original columns not encoded
car_x = np.empty((clean_car_data.shape[0] - 1, clean_car_data.shape[1] - 1))
attributes = clean_car_data[1:, :-1]

    # Function to encode a column and get new column names

for i , column in enumerate(attributes.T):
    new_column = np.zeros(len(column))
    categories = np.unique(column)
    categories_dict = {}

    for category in categories:
        categories_dict[category] = len(categories_dict)

    for j, val in enumerate(column):
        new_column[j] = int(categories_dict[val])


    car_x[:,i] = new_column

car_y = np.zeros((clean_car_data.shape[0] - 1, 1))

for i, val, in enumerate(clean_car_data[1:, -1]):
    if val == 'unacc':
        car_y[i] = 0

    else:
        car_y[i] = 1

####################################### ionosphere #######################################
filename = os.path.join(script_directory, 'ionosphere_num.npy')

ionosphere_data = np.load(filename, allow_pickle=True, fix_imports=True)  # Load data from .npy file


missing_rows = [i for i in range(ionosphere_data.shape[0]) if '?' in ionosphere_data[i]]
clean_ionosphere_data = np.delete(ionosphere_data, missing_rows, axis=0) # Delete rows with missing data
column_names = clean_ionosphere_data[0]

# Prepare the new dataset with the original columns not encoded
ionosphere_x = clean_ionosphere_data[1:,:-1]

ionosphere_y = np.zeros((clean_ionosphere_data.shape[0] - 1, 1))

for i, val, in enumerate(clean_ionosphere_data[1:, -1]):
    if val == 'b':
        ionosphere_y[i] = 0

    else:
        ionosphere_y[i] = 1


####################################### iris #######################################

filename = os.path.join(script_directory, 'iris_num.npy')

iris_data = np.load(filename, allow_pickle=True, fix_imports=True)  # Load data from .npy file

missing_rows = [i for i in range(iris_data.shape[0]) if '?' in iris_data[i]]
clean_iris_data = np.delete(iris_data, missing_rows, axis=0) # Delete rows with missing data
column_names = clean_iris_data[0]

# Prepare the new dataset with the original columns not encoded
iris_x = clean_iris_data[1:,:-1]

iris_y = np.zeros((clean_iris_data.shape[0] - 1, 1))

for i, val, in enumerate(clean_iris_data[1:, -1]):
    if val == 'Iris-setosa':
        iris_y[i] = 1

    else:
        iris_y[i] = 0


knn1 = KNN(1)
knn2 = KNN(2)
knn3 = KNN(3)
knn4 = KNN(4)
knn5 = KNN(5)
knn6 = KNN(6)
knn7 = KNN(7)

k_fold_cross_validation(knn3, adult_x, adult_y, 5)
