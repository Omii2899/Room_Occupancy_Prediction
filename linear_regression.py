import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LinearRegression:
    """
    Linear Regression : This class performs linear regression

    Attributes
    __________
    iterations : number of iterations used for gradient descent to find optimal weights,
    learning_rate : learning rate used for gradient descent(alpha),
    tolerance : tolerance used for gradient descent convergence of difference in weights,
    solution_type : type of solution(Gradient Descent = 'gd' , Stochastic Gradient Descent = 'sgd' , Closed Form
    Solution = 'closed') Default = 'gd',
    l1_regularization: True for L1 regularization Default = 'False',
    l2_regularization: True for L2 regularization Default  = 'False',
    regularization_constant : regularization constant(lambda), Default = 0,
    mean : mean of the data used to fit the model,
    std : standard deviation of the data used to fit the model,
    low_rank : If the data used to fit the model is low-rank then Closed Form Solution is not possible,
    full_rank : If the data used to fit the model is full-rank then Closed Form Solution is possible,
    weights : weights of the model after fitting,
    errors : errors of the model in each iteration of gradient descent

    Methods
    _______
    fit : fit the data to the model,
    predict : predict the test data using fitted model weights,
    evaluate : predict the test data using fitted model weights and get evaluation results RMSE,SSE
    """

    def __init__(self, iterations, learning_rate, tolerance, solution_type='gd', l1_regularization=False,
                 l2_regularization=False, regularization_constant=0, ):
        """
        This is the constructor of the object for class LinearRegression
        :param iterations: Number of iterations used for gradient descent to find optimal weights
        :param learning_rate:  learning rate used for gradient descent(alpha)
        :param tolerance: tolerance used for gradient descent convergence for difference in weights
        :param solution_type: type of solution(Gradient Descent = 'gd' , Stochastic Gradient Descent = 'sgd' ,
            Closed Form Solution = 'closed'), Default = 'gd'
        :param l1_regularization: True for L1 regularization, Default = 'False'
        :param l2_regularization: True for L2 regularization, Default  = 'False'
        :param regularization_constant: regularization constant(lambda), Default = 0
        """
        self.y_hat = None
        self.low_rank = None
        self.full_rank = None
        self.X_test = None
        self.std = None
        self.X_train = None
        self.mean = None
        self.weights = None
        self.tolerance = tolerance
        self.regularization_constant = regularization_constant
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.solution_type = solution_type
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.regularization_constant = regularization_constant

    def _normalize_train_data(self, X_train):
        """
        Normalize the training data and saves mean and standard deviation in the instance
        :param X_train: Data to be normalized
        :return:  input data with mean and standard deviation
        """
        self.mean = np.mean(X_train, axis=0)
        self.std = np.std(X_train, axis=0)
        self.X_train = (X_train - self.mean) / self.std
        return self.X_train

    def _normalize_test_data(self, X_test):
        """
        Normalize the test data using the same mean and standard deviation used to normalize the training data
        :param X_test: Test data to be normalized
        :return: Normalized Test data with mean and standard deviation
        """
        if self.mean or self.std:
            self.X_test = (X_test - self.mean) / self.std
            return self.X_test
        else:
            raise ValueError("No mean and Std, Normalize training data first")

    @staticmethod
    def _add_column_zero(self, X):
        """
        Adds the column of bias to the input data
        :param X: Input Data(Numpy Array)
        :return: Data with bias column of 1's as first column
        """
        X = np.column_stack((np.ones((X.shape[0], 1)), X))
        return X

    def _full_rank_check(self, X_train):
        """
        Checks if the input data has a full rank or not and assigns a boolean to full_rank variable of the instance
        :param X_train: Input Training Data(Numpy Array)
        :return: True if the input data has a full rank
        """
        rank = np.linalg.matrix_rank(X_train)
        if rank == min(X_train.shape):
            self.full_rank = True
        else:
            self.full_rank = False

        return self.full_rank

    def _low_rank_check(self, X_train):
        """
        Checks if the input data has a low rank or not and assigns a boolean to low_rank variable of the instance
        :param X_train: Input Training Data(Numpy Array)
        :return: True if the input data has a low rank
        """
        if X_train.shape[0] < X_train.shape[1]:
            self.low_rank = True
        else:
            self.low_rank = False
        return self.low_rank

    def _closed_solution(self, X_train, y_train):
        """
        Gives the closed solution answer by calculating the optimum weights using closed form formula.
        Only Applicable for full rank data
        Solutions available for Linear regression and Ridge regression(l2_regularization)
        :param X_train: Input Training Data(Numpy Array)
        :param y_train: Input Target Values (Numpy Array)
        :return: None
        """
        if self.l2_regularization:
            if self._full_rank_check(X_train) and not self._low_rank_check(X_train):
                self.weights = np.matmul(np.linalg.inv(
                    np.matmul(X_train.T, X_train) + self.regularization_constant * np.identity(X_train.shape[1])),
                    (np.matmul(X_train.T, y_train)))
            else:
                print("Matrix is low rank")
                return
        elif self.l1_regularization:
            print('Solution Not Available for l1_regularization(LASSO Regression)')
            return
        else:
            if self._full_rank_check(X_train) and not self._low_rank_check(X_train):
                self.weights = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T.dot(y_train))
            else:
                print("Matrix is low rank")
                return

    @staticmethod
    def _sse_function(self, y_hat, y):
        """
        Calculates the squared error between the prediction and the true label
        :param y_hat: Predicted Target
        :param y: True target
        :return: loss
        """
        loss = np.sum(np.square(y_hat - y))
        return loss

    @staticmethod
    def _rmse_function(self, y_hat, y):
        """
        Calculates the root mean squared error between the prediction and the true Target
        :param y_hat: Predicted Target
        :param y: True target
        :return: RMSE
        """
        return np.sqrt(self._sse_function(y_hat, y) / y_hat.shape[0])

    def _cost_derivative(self, X, y):
        """
        Calculates the cost derivative for Gradient Descent
        :param X: Input Features(Numpy Array) 
        :param y: True target
        :return: Cost Derivative
        """
        return np.matmul(X.T, (np.matmul(X, self.weights) - y))

    def _gradient_descent(self, X_train, y_train):
        """
        Calculates the weights using gradient descent
        :param X_train: Training Input array
        :param y_train: Training target Array
        :return: None
        """
        prev_error = np.inf
        errors = list()
        if self.regularization:
            for i in tqdm(range(1, self.iterations), colour='MAGENTA'):
                self.weights -= self.learning_rate * (
                        self._cost_derivative(X_train, y_train) + (self.regularization_constant * self.weights))
                y_hat = np.matmul(X_train, self.weights)
                error = self._rmse_function(y_hat, y_train)
                errors.append(error)
                if np.abs(error - prev_error) < self.tolerance:
                    print("Model Stopped learning")
                    break
                prev_error = error
        else:
            for i in tqdm(range(1, self.iterations), colour='MAGENTA'):
                self.weights -= self.learning_rate * self._cost_derivative(X_train, y_train)
                error = self._rmse_function(X_train, y_train)
                errors.append(error)
                if np.abs(error - prev_error) < self.tolerance:
                    print("Model Stopped learning")
                    break
                prev_error = error

    def fit(self, X_train, y_train) -> None:
        """
        Fits the model to the training data after normalizing it using mean and standard deviation
        :param X_train: Training data
        :param y_train: Training target variable
        :return: None
        """
        X_train = self._normalize_train_data(X_train)
        X_train = self._add_column_zero(X_train)
        if self.solution_type == 'closed':
            self._closed_solution(X_train, y_train)
        elif self.solution_type == 'gd':
            self.weights = np.zeros(X_train.shape[1])
            self._gradient_descent(X_train, y_train)
        elif self.solution_type == 'sgd':
            self.weights = np.zeros(X_train.shape[1])
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
            self._gradient_descent(X_train, y_train)

    def predict(self, X_test) -> np.ndarray:
        """
        Predicts the value of Target Variable for given input data
        :param X_test: Input Test Data
        :return: Predicted Target
        """
        X_test = self._normalize_test_data(X_test)
        X_test = self._add_column_zero(X_test)
        self.y_hat = np.matmul(X_test, self.weights)
        return self.y_hat

    def evaluate(self, X, y, mode) -> None:
        """
        Evaluates the performance of the model on the given input data, targets and mode
        :param X: Input Data
        :param y: Input Target Data
        :param mode: 'train' or 'test'
        :return: None
        """
        if mode == 'train':
            results = self.predict(X)
            sse = self._sse_function(results, y)
            rmse = self._rmse_function(results, y)
            print('Train set RMSE: ', rmse)
            print('Train set SSE: ', sse)
        elif mode == 'test':
            results = self.predict(X)
            sse = self._sse_function(results, y)
            rmse = self._rmse_function(results, y)
            print('Test set RMSE: ', rmse)
            print('Test set SSE: ', sse)
