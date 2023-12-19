# Import

import numpy as np

# Naice non-parametric bayes classifier

class bayes_nonparametric:
    '''
    Naive non-parametric Bayes classifier.

    This class implements a simple non-parametric Bayes classifier based on kernel density estimation.
    '''

    def __init__(self, h):
        '''Initialize model.
        
        Params:
            h (float): Bandwidth parameter for kernel density estimation.
        '''
        self.h = h # Bandwidth

    def fit(self, X_train, y_train):
        '''
        Train the classifier with the given training data.

        Params:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            h (float): Bandwidth parameter for kernel density estimation.
        '''
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
        
    def fit_kde(self, x):
        '''Kernel density estimate for given input point x.'''

        # Initialize multivariate feature distribution for each class
        multivariate_estimates = {cl: 1 for cl in self.classes}

        # Iterate over each class
        for cl in self.classes:

            class_data = self.X_train[self.y_train == cl] # Data in X_train belonging to class
            n_total = class_data.shape[0] # Number of observations in class

            # Iterate over each feature in class_data
            for i, X_i in enumerate(class_data.T):

                # Count observations in bandwith range from x
                # n_observations = sum(1 for obs in X_i if np.abs(x - obs) <= self.h)
                n_observations = np.sum(np.abs(X_i - x[i]) <= self.h)

                # Estimate univariate feature distribution for feature X
                kernel_estimate = n_observations / (n_total * 2 * self.h)

                # Update the multivariate feature distribution estimate
                multivariate_estimates[cl] *= kernel_estimate

        return multivariate_estimates

    def get_posterior_probabilities(self, x):
        '''Calculate posterior probabilities for each class for an input point x.'''

        # Compute likelihoods
        likelihoods = self.fit_kde(x)

        # Compute evidence for each class
        evidence = sum(likelihoods[cl] * self.priors[cl] for cl in self.classes)
        if evidence == 0: # In case of evidence is zero, add small constant
            evidence += 1e-12 

        # Compute posterior probabilities for each class
        posteriors = {cl: likelihoods[cl] * self.priors[cl] / evidence for cl in self.classes}

        return posteriors
    
    def _predict(self, x):
        '''Predict the class for a single data point x.'''
        # print('Shape of single point x:', x.shape)

        # Compute posterior probabilities for each class
        posteriors = self.get_posterior_probabilities(x)

        # Return class with highest posterior probabilities
        return max(posteriors, key = posteriors.get)
    
    def predict(self, X):
        '''Predict the classes for an input array X.'''
        return np.array([self._predict(x) for x in X])