# Imports

import numpy as np

# LDA class

class LDA:
    def __init__(self, k = 2):
        self.k = k # Amount of discriminant variables
        
    def fit_transform(self, X, y):

        class_labels = np.unique(y)
        m = X.shape[1] # Amount of features

        # Compute total mean of all samples
        mu = np.mean(X, axis = 0).reshape(m, 1) # Reshape to be vertical vector

        # Initiate within- and between-class scatter matrices as empty
        S_W = np.zeros((m, m))
        S_B = np.zeros((m, m))

        # Compute within- and between-class scatter matrices by iterating over each class i
        for label in class_labels:

            X_i = X[y == label] # Matrix of data points for class i
            mu_i = np.mean(X_i, axis = 0).reshape(m, 1) # Sample mean of class i

            # Initiate withing scatter matrix for class i as empty
            S_i = np.zeros((m, m))
            for row in X_i:
                row = row.reshape(m, 1) # Reshape row to be vertical vector
                S_i += np.dot(row - mu_i, (row - mu_i).T)

            # Add scatter matrix of class i to within-class scatter matrix
            S_W += S_i

            # Update between-class scatter matrix using amount of samples, as well as class and total means
            n_i = X_i.shape[0] # Amount of samples in class i
            S_B += n_i * np.dot(mu_i - mu, (mu_i - mu).T)

        # Compute the inverse of the within-class scatter matrix
        S_W_inv = np.linalg.inv(S_W)

        # Solve the generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(S_W_inv, S_B))
        
        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top k eigenvectors
        selected_eigenvectors = eigenvectors[:, :self.k]

        # Make sure numbers in eigenvectors are not complex
        selected_eigenvectors = np.real(selected_eigenvectors)

        # Normalize eigenvectors
        normalized_eigenvectors = selected_eigenvectors / np.linalg.norm(selected_eigenvectors, axis = 0)

        # Project the data onto the normalized discriminant subspace
        X_projected = np.dot(X, normalized_eigenvectors)

        return X_projected