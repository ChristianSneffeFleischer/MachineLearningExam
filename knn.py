import numpy as np

class KNN:
    def __init__(self, n_neighbors):
        '''Initialize model.
        
        Params:
            n_neighbors (int): K-nearest neighbors parameter.
        '''
        self.n_neighbors = n_neighbors # (k) Number of closest neighbors to base classification on

    def fit(self, X_train, y_train):
        '''
        Train the classifier with the given training data.

        Params:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            h (float): Bandwidth parameter for kernel density estimation.
        '''
        # Store training data
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

        # Calculate class prior probailities
        self.calculate_class_priors()

    def calculate_class_priors(self):
        '''Calculate class prior probabilities from the training data.'''
        class_counts = {cl: np.sum(self.y_train == cl) for cl in self.classes}
        n_samples = len(self.y_train)
        self.priors = {cl: count/n_samples for cl, count in class_counts.items()}

    def get_neighbor_labels(self, x):
        '''Get the labels of the k nearest neighbors to input x.'''

        # Compute distance between x and all observations in the training set
        distances = [np.linalg.norm(x - obs) for obs in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        return k_nearest_labels

    def _predict(self, x):
        '''Predict the class for a single data point x.'''

        # Compute classes for the selected amount of nearest neighbors
        k_nearest_labels = self.get_neighbor_labels(x)

        # Calculate weighted votes based on amount of neighbors in each class
        weighted_votes = {cl: k_nearest_labels.count(cl) / self.n_neighbors for cl in self.classes}

        # Return highest weighted vote class
        return max(weighted_votes, key = weighted_votes.get)
    
    def predict(self, X):
        '''Predict the classes for an input array X.'''
        return np.array([self._predict(x) for x in X])
