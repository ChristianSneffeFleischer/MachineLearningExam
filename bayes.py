# Import

import numpy as np

# Naice non-parametric bayes classifier

class bayes_nonparametric:
    '''
    Naive non-parametric Bayes classifier.

    This class implements a simple non-parametric Bayes classifier based on kernel density estimation.
    '''

    def __init__(self):
        pass

    def train(self, X_train, y_train, h):
        '''
        Train the classifier with the given training data.

        Params:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            h (float): Bandwidth parameter for kernel density estimation.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.h = h # Bandwidth
        self.classes = np.unique(y_train)

        # Calculate class prior probailities
        self.calculate_class_priors()

        # Calculate kernel density estimates for each class
        self.kde_estimators = {k: self.fit_kde(X_train[y_train == k]) for k in self.classes}

    def calculate_class_priors(self):
        '''Calculate class prior probabilities from the training data.'''
        class_counts = {k: np.sum(self.y_train == k) for k in self.classes}
        n_samples = len(self.y_train)
        self.priors = {k: count/n_samples for k, count in class_counts.items()}

    # def fit_kde(self, data):
    #     '''Fit a kernel density estimate to the given input data.'''
        
    #     def kernel_density_estimate(X, data):
    #         '''Kernel density estimate for given input point or array X.'''
    #         n = len(data) # Number of observations in the input data
    #         kernel_vals = np.ones(X.shape) # Initialize array for the kernel values

    #         # Iterate over each data point
    #         for obs in data:
    #             # Update kernel value based on the absolute difference between the current observation and X
    #             kernel_vals -= np.abs(X - obs) > self.h
    #         return np.sum(kernel_vals, axis = 0) / (n * 2 * self.h)
        
    #     vectorized_estimates = np.vectorize(lambda X: kernel_density_estimate(X, data))
    #     return vectorized_estimates
    
    # def get_likelihoods(self, x):
    #     '''Get likelihood for each class for an input point x.'''
    #     likelihoods = {k: self.kde_estimators[k](x) for k in self.classes}
    #     return likelihoods
        
    def fit_kde(self, x):
        '''Kernel density estimate for given input point x.'''

        print(x.shape)

        # Initialize multivariate feature distribution for each class
        multivariate_estimates = {k: 1 for k in self.classes}

        # Iterate over each class
        for k in self.classes:

            X_k = self.X_train[self.y_train == k] # Data in X_train belonging to class k
            n = X_k.shape[0] # Number of observations in class k

            # Iterate over each feature in X_k
            for X in X_k.T:

                # Count observations in bandwith range from x
                count = sum(1 for obs in X if np.abs(x - obs) <= self.h)

                # Estimate univariate feature distribution for feature X
                kernel_estimate = count / (n * 2 * self.h)

                # Update the multivariate feature distribution estimate
                multivariate_estimates[k] *= kernel_estimate

        return multivariate_estimates

    def get_posterior_probabilities(self, x):
        '''Calculate posterior probabilities for each class for an input point x.'''

        # Compute likelihoods
        likelihoods = self.fit_kde(x)

        # Compute evidence for each class
        evidence = sum(likelihoods[k] * self.priors[k] for k in self.classes)

        # Compute posterior probabilities for each class
        posteriors = {k: likelihoods[k] * self.priors[k] / evidence for k in self.classes}

        return posteriors
    
    def _predict(self, x):
        '''Predict the class for a single data point x.'''
        # print('Shape of single point x:', x.shape)

        # Compute posterior probabilities for each class
        posteriors = self.get_posterior_probabilities(x)

        # Return class with highest posterior probabilities
        return min(posteriors, key = posteriors.get)
    
    def predict(self, X):
        '''Predict the class for an input array X.'''
        # print('Shape of input X:', X.shape)
        return [self._predict(x) for x in X]