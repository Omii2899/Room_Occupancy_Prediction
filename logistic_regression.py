import numpy as np
from tqdm import tqdm


class LogisticRegression:
    """
    Logistic Regression Class

    Attributes
    __________
    iterations : number of iterations used for gradient descent to find optimal weights,
    learning_rate : learning rate used for gradient descent(alpha),
    tolerance : tolerance used for gradient descent convergence of difference in weights,
    regularization_parameter : regularization parameter(lambda), Default = 0,
    weights : weights of the model after fitting,
    errors : errors of the model in each iteration of gradient descent

    Methods
    _______
    fit(X_train, y_train) : train model on training set
    predict(X_test) : predict the test data using fitted model weights,
    evaluate(X_test,y_test) : predict the test data using fitted model weights and get evaluation results Accuracy,
    Recall, F1 Score, Precision
    """
    def __init__(self, iterations, learning_rate, tolerance, regularization=None, regularization_parameter=0):
        self.errors = None
        self.prev_cost = None
        self.cost = None
        self.mean = None
        self.std = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.weights = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.regularization = regularization
        self.regularization_parameter = regularization_parameter

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def gradient_calculation(self):
        """
        Calculates the gradient of the logistic regression model's loss function with respect to the model's weights.
        The gradient calculation formula is:

        ∇(loss) = X.T * (y_pred - y)
        :return:
        numpy.ndarray: A gradient vector where each element is the partial
                       derivative of the loss function with respect to the
                       corresponding weight.
        """
        y_pred = self.sigmoid(np.matmul(self.X_train, self.weights))
        return self.X_train.T.dot(y_pred - self.y_train)

    def cost_function(self):
        """
        Computes the cost for a logistic regression model using cross-entropy loss
        and adds regularization penalty if applicable.

        The cost function is defined as:

            L = -sum(y * log(y_pred) + (1 - y) * log(1 - y_pred)) + penalty

        where:
        - y is the vector of actual class labels,
        - y_pred is the vector of predicted probabilities, computed as sigmoid(X * w),
        - penalty is the regularization term, which can be L1 (sum of absolute values of weights)
          or L2 (sum of squares of weights), depending on the regularization type specified.

        Returns:
            float: The computed cost value.
        """
        y_pred = self.sigmoid(np.matmul(self.X_train, self.weights))
        cost = self.y_train * np.log(y_pred) + (1 - self.y_train) * np.log(1 - y_pred)
        penalty = 0
        if self.regularization == 'l1':
            penalty = np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            penalty = np.sum(np.square(self.weights))
        cost = -np.sum(cost) + penalty
        return cost

    def gradient_descent(self):
        """
        Performs gradient descent optimization to find the optimal weights of the logistic regression model.

        The method iteratively updates the weights by moving in the direction of the negative gradient of the cost function, adjusted by the learning rate. It optionally includes L1 or L2 regularization:

        Weight update formulas:
        - Without regularization: w = w - learning_rate * ∇(loss)
        - With L1 regularization: w = w - learning_rate * ∇(loss) - regularization_parameter
        - With L2 regularization: w = w - learning_rate * ∇(loss) - regularization_parameter * w

        The process continues for a specified number of iterations or until the improvement in cost is less than a defined tolerance, indicating convergence.

        Attributes updated during optimization:
        - self.weights: The weights vector of the model.
        - self.errors: A list of cost values at each iteration, tracking how the cost changes.

        Side effects:
        - If the change in cost between iterations is less than the tolerance, it prints a message and stops further updates.

        """
        self.errors = []
        self.weights = np.zeros(shape=(self.X_train.shape[1], 1))
        self.prev_cost = np.inf
        for i in tqdm(range(self.iterations), colour='WHITE'):
            if self.regularization == 'l1':
                self.weights -= ((self.learning_rate * self.gradient_calculation()) +
                                 (self.regularization_parameter * np.ones(shape=(self.X_train.shape[1], 1))))
            elif self.regularization == 'l2':
                self.weights -= ((self.learning_rate * self.gradient_calculation()) +
                                 (self.regularization_parameter * self.weights))
            else:
                self.weights -= (self.learning_rate * self.gradient_calculation())
            self.cost = self.cost_function()
            self.errors.append(self.cost)
            if self.prev_cost - self.cost <= self.tolerance:
                print('Model has stopped improving')
                break
            self.prev_cost = self.cost

    def fit(self, X_train, y_train):
        """
        Fits the logistic regression model to the training data using gradient descent.

        This method initializes the training process by setting the training data and target labels,
        and then calls the `gradient_descent` method to optimize the model's weights.

        Parameters:
            X_train (array-like): The input features of the training data.
            y_train (array-like): The target labels corresponding to the input features.

        The training process involves:
        - Storing the training data (`X_train`) and labels (`y_train`) in the instance variables.
        - Calling the `gradient_descent` method to adjust the weights based on the loss gradient.

        Note:
        - The shape of `X_train` should match the expected number of features.
        - The shape of `y_train` should correspond to the number of samples in `X_train`.
        """

        self.X_train = X_train
        self.y_train = y_train
        self.gradient_descent()

    def predict(self, X_test):
        """
        Predicts class labels for given input samples using the trained logistic regression model.

        This method computes predictions by applying the sigmoid function to the linear combination
        of input features and the learned weights. The output of the sigmoid function represents
        the probability of the input belonging to the positive class, which is then thresholded at 0.5
        to produce binary class labels.

        Parameters:
            X_test (array-like): The input features of the test data.

        Returns:
            numpy.ndarray: An array of predicted class labels (0 or 1) for each input sample.

        The prediction is calculated as follows:
        - Apply the sigmoid function to (X_test * weights) to get the probability of the positive class.
        - Threshold the probabilities at 0.5 to determine the class labels.
        """
        predictions = self.sigmoid(np.matmul(X_test, self.weights))
        return np.round(predictions)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the logistic regression model on a test dataset.

        This method uses the model's predictions to calculate key classification metrics:
        accuracy, precision, recall, and F1-score. These metrics provide insights into the
        effectiveness of the model in classifying positive and negative classes.

        Parameters:
            X_test (array-like): The input features of the test data.
            y_test (array-like): The actual class labels for the test data.

        The evaluation metrics are calculated as follows:
        - True Positives (TP): Correct positive predictions.
        - False Positives (FP): Incorrect positive predictions.
        - True Negatives (TN): Correct negative predictions.
        - False Negatives (FN): Incorrect negative predictions.

        Formulas:
        - Accuracy = TP / (TP + FP)
        - Recall = TP / (TP + FN)
        - Precision = TP / (TP + FP)
        - F1-score = 2 * (Precision * Recall) / (Precision + Recall)

        Side effects:
        - Prints the calculated metrics: Accuracy, Recall, Precision, and F1-score.
        """
        y_predicted = self.predict(X_test)
        true_positives = np.sum(y_predicted * y_test)
        false_positive = np.sum(y_predicted * (1 - y_test))
        true_negative = np.sum((y_predicted == 0) & (y_test == 0))
        false_negative = np.sum((y_predicted == 0) & (y_test == 1))

        accuracy = true_positives / (true_positives + false_positive)
        recall = true_positives / (true_positives + false_negative)
        precision = true_positives / (true_positives + false_positive)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print('Accuracy: ', accuracy)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F1-score: ', f1_score)
