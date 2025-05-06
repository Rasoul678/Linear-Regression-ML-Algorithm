import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Linear Regression model.

        Parameters:
        learning_rate (float): The step size at each iteration (default: 0.01)
        n_iterations (int): Number of iterations for gradient descent (default: 1000)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # Coefficients for each feature
        self.bias = None  # Intercept term

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.

        Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
        y (numpy.ndarray): Target values of shape (n_samples,)
        """
        # Get number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent optimization
        for i in range(self.n_iterations):
            # Predict y values with current weights and bias
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            # Gradient of weights: (1/n_samples) * X^T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # Gradient of bias: (1/n_samples) * sum(y_pred - y)
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias:
            # w = w - a * dw
            # b = b - A * db
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)

        Returns:
        numpy.ndarray: Predicted values
        """
        return np.dot(X, self.weights) + self.bias

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Compute mean squared error between true and predicted values.

        Parameters:
        y_true (numpy.ndarray): Actual target values
        y_pred (numpy.ndarray): Predicted target values

        Returns:
        float: Mean squared error
        """
        return np.mean((y_true - y_pred) ** 2)