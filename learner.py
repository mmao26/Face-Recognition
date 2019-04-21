### Author Name: Manqing Mao
### GTID: mmao33

"""PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    
    """
    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    imgs = [np.array(cv2.imread(os.path.join(folder, f), 0)) for f in images_files]
    imgs = [cv2.resize(x, size) for x in imgs]
    imgs_flatten = [x.flatten() for x in imgs]

    X = np.empty([len(images_files), size[0]*size[1]])
    y = np.empty(len(images_files), dtype=int)

    label_dict = {'01': int(1), '02': int(2), '03': int(3), '04': int(4), '05': int(5),
                 '06': int(6), '07': int(7), '08': int(8), '09': int(9), '10': int(10),
                 '11': int(11), '12': int(12), '13': int(13), '14': int(14), '15': int(15)}
 
    for i in range(len(images_files)):
        X[i,:] = imgs_flatten[i]
        y[i] = label_dict[images_files[i][7:9]]

    return X, y


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    
    # num_images
    M = X.shape[0]
    N = int(round(M * p))
    
    indices = np.random.permutation(M)
    train_idx, test_idx = indices[:N], indices[N:]
    Xtrain, Xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]

    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    X_temp = np.copy(X)
    X_mean = get_mean_face(X_temp)
    delta = X_temp - X_mean
    # w -- array: eigenvalues
    # v -- matrix: eigenvectors
    sigma = np.dot(delta.T, delta)
    w, v = np.linalg.eigh(sigma)
    # pick last k values
    eigenvalues = w[-k:]
    eigenvectors = v[:, -k:]
    # reorder
    eigenvalues = np.flip(eigenvalues, 0)
    eigenvectors = np.flip(eigenvectors, 1)
    
    return eigenvectors, eigenvalues


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for j in range(self.num_iterations):
            # a) Renormalize the weights so they sum up to 1
            self.weights = self.weights / np.sum(self.weights)
            
            # b) Instantiate the weak classifier h with the training data and labels
            weak_h = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            #    Train the classifier h
            weak_h.train()
            #    Weak classifiers appended in each iteration.
            self.weakClassifiers.append(weak_h)
            #    Get predictions h(x) for all training examples
            h_x = np.array([weak_h.predict(x) for x in self.Xtrain], np.float32)
            
            # c) Find epsilon_j = sum(weights) where h_x ~= y
            epsilon_j = np.sum(np.multiply(self.weights, np.float32(np.not_equal(h_x, self.ytrain))))
            # d) Calculate alpha_j
            alpha_j = - 0.5 * np.log(epsilon_j / (1 - epsilon_j)) 
            self.alphas.append(alpha_j)
            # e)
            if epsilon_j > self.eps:
                exp_j = -np.multiply(self.ytrain, alpha_j * h_x) 
                self.weights = np.multiply(self.weights, np.exp(exp_j))
            else:
                break
            
    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        ypredicted_train = self.predict(self.Xtrain)
        incorrect = int(np.sum(np.float64(np.not_equal(ypredicted_train, self.ytrain))))
        correct = len(ypredicted_train) - incorrect
        return correct, incorrect


    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        # Each col represents an image data
        predict_weights = np.empty([len(self.weakClassifiers), X.shape[0]], np.float32)
        # H(x) = sign[sum(alpha * h_x)]
        for i in range(len(self.weakClassifiers)):
            # use each weak classifier to predict X and store in a row
            predict_weights[i, :] = self.alphas[i] * np.array([self.weakClassifiers[i].predict(x_temp) for x_temp in X], np.float32)

        y_predict = np.sum(predict_weights, axis=0, dtype=np.float32)
        y_predict = np.float32(y_predict > 0) * 2.0 - 1.0

        return y_predict


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height = self.size[0]
        width = self.size[1]
        half_height = np.divide(int(height), 2)

        # set all black
        feat_2_1 = np.zeros(shape)
        # set white
        feat_2_1[self.position[0] : self.position[0]+half_height, self.position[1] : self.position[1]+width] = 255
        # set gray
        feat_2_1[self.position[0]+half_height : self.position[0]+height, self.position[1] : self.position[1]+width] = 126

        return feat_2_1

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height = self.size[0]
        width = self.size[1]
        half_width = np.divide(int(width), 2)
        
        # set all black
        feat_1_2 = np.zeros(shape)
        # set white
        feat_1_2[self.position[0] : self.position[0]+height, self.position[1] : self.position[1]+half_width] = 255
        # set gray
        feat_1_2[self.position[0] : self.position[0]+height, self.position[1]+half_width : self.position[1]+width] = 126

        return feat_1_2

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height = self.size[0]
        width = self.size[1]
        one_third_height = np.divide(int(height), 3)

        # set all black
        feat_3_1 = np.zeros(shape)
        # set the first strip white
        feat_3_1[self.position[0] : self.position[0]+one_third_height, self.position[1] : self.position[1]+width] = 255
        # set the second strip gray
        feat_3_1[self.position[0]+one_third_height : self.position[0]+2*one_third_height, self.position[1] : self.position[1]+width] = 126
        # set the third strip white
        feat_3_1[self.position[0]+2*one_third_height : self.position[0]+height, self.position[1] : self.position[1]+width] = 255

        return feat_3_1

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        height = self.size[0]
        width = self.size[1]
        one_third_width = np.divide(int(width), 3)
        
        # set all black
        feat_1_3 = np.zeros(shape)
        # set the first strip white
        feat_1_3[self.position[0] : self.position[0]+height, self.position[1] : self.position[1]+one_third_width] = 255
        # set the second strip gray
        feat_1_3[self.position[0] : self.position[0]+height, self.position[1]+one_third_width : self.position[1]+2*one_third_width] = 126
        # set the third strip white
        feat_1_3[self.position[0] : self.position[0]+height, self.position[1]+2*one_third_width : self.position[1]+width] = 255

        return feat_1_3

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        height = self.size[0]
        width = self.size[1]
        half_height = np.divide(int(height), 2)
        half_width = np.divide(int(width), 2)
        
        # set all black
        feat_2_2 = np.zeros(shape)
        # set the top left square gray
        feat_2_2[self.position[0] : self.position[0]+half_height, self.position[1] : self.position[1]+half_width] = 126
        # set the top right square white
        feat_2_2[self.position[0]+half_height : self.position[0]+height, self.position[1] : self.position[1]+half_width] = 255
        # set the bottom left square white
        feat_2_2[self.position[0] : self.position[0]+half_height, self.position[1]+half_width : self.position[1]+width] = 255
        # set the bottom right square gray
        feat_2_2[self.position[0]+half_height : self.position[0]+height, self.position[1]+half_width : self.position[1]+width] = 126

        return feat_2_2

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        ii_temp = np.copy(ii)
        
        height = self.size[0]
        width = self.size[1]
        half_height = np.divide(int(height), 2)
        half_width = np.divide(int(width), 2)
        one_third_height = np.divide(int(height), 3)
        one_third_width = np.divide(int(width), 3)
        
        # sum(D) = ii(4) - ii(2) - ii(3) + ii(1)
        if self.feat_type == (2, 1):

            sum_white_area = ii_temp[self.position[0] + half_height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + half_height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] - 1]

            sum_grey_area = ii_temp[self.position[0] + height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + half_height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + height - 1, self.position[1] - 1] + \
                            ii_temp[self.position[0] + half_height - 1, self.position[1] - 1]

        elif self.feat_type == (1, 2):

            sum_white_area = ii_temp[self.position[0] + height - 1, self.position[1] + half_width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + half_width - 1] - \
                             ii_temp[self.position[0] + height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] - 1]

            sum_grey_area = ii_temp[self.position[0] + height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + height - 1, self.position[1] + half_width - 1] + \
                            ii_temp[self.position[0] - 1, self.position[1] + half_width - 1]

        elif self.feat_type == (3, 1):

            sum_white_area = ii_temp[self.position[0] + one_third_height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + one_third_height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] + height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + 2 * one_third_height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] + 2 * one_third_height - 1, self.position[1] - 1]

            sum_grey_area = ii_temp[self.position[0] + 2 * one_third_height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + one_third_height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + 2 * one_third_height - 1, self.position[1] - 1] + \
                            ii_temp[self.position[0] + one_third_height - 1, self.position[1] - 1]

        elif self.feat_type == (1, 3):

            sum_white_area = ii_temp[self.position[0] + height - 1, self.position[1] + one_third_width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + one_third_width - 1] - \
                             ii_temp[self.position[0] + height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] + height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + height - 1, self.position[1] + 2 * one_third_width - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] + 2 * one_third_width - 1]

            sum_grey_area = ii_temp[self.position[0] + height - 1, self.position[1] + 2 * one_third_width - 1] - \
                            ii_temp[self.position[0] - 1, self.position[1] + 2 * one_third_width - 1] - \
                            ii_temp[self.position[0] + height - 1, self.position[1] + one_third_width - 1] + \
                            ii_temp[self.position[0] - 1, self.position[1] + one_third_width - 1]

        elif self.feat_type == (2, 2):

            sum_white_area = ii_temp[self.position[0] + height - 1, self.position[1] + half_width - 1] - \
                             ii_temp[self.position[0] + half_height - 1, self.position[1] + half_width - 1] - \
                             ii_temp[self.position[0] + height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] + half_height - 1, self.position[1] - 1] + \
                             ii_temp[self.position[0] + half_height - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] - 1, self.position[1] + width - 1] - \
                             ii_temp[self.position[0] + half_height - 1, self.position[1] + half_width - 1] + \
                             ii_temp[self.position[0] - 1, self.position[1] + half_width - 1]

            sum_grey_area = ii_temp[self.position[0] + half_height - 1, self.position[1] + half_width - 1] - \
                            ii_temp[self.position[0] - 1, self.position[1] + half_width - 1] - \
                            ii_temp[self.position[0] + half_height - 1, self.position[1] - 1] + \
                            ii_temp[self.position[0] - 1, self.position[1] - 1] + \
                            ii_temp[self.position[0] + height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + half_height - 1, self.position[1] + width - 1] - \
                            ii_temp[self.position[0] + height - 1, self.position[1] + half_width - 1] + \
                            ii_temp[self.position[0] + half_height - 1, self.position[1] + half_width - 1]

        Score = float(sum_white_area) - float(sum_grey_area)
        return Score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    images_integralList = []
    for img in images:
        image_integral = np.zeros(img.shape)
        sum_row = np.zeros(img.shape)
        for col in range(img.shape[1]):
            for row in range(img.shape[0]):
                # get the integral in row direction
                sum_row[row, col] = sum_row[row - 1, col] + img[row, col]
                # get the integral in column direction
                image_integral[row, col] = image_integral[row, col - 1] + sum_row[row, col]
        images_integralList.append(image_integral)

    return images_integralList


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"
        for i in range(num_classifiers):

            # TODO: Complete the Viola Jones algorithm
            # 1. Normalize the weights
            weights = weights / np.sum(weights)
            # 2. Instantiate a and train a classifier h_j
            h_j = VJ_Classifier(scores, self.labels, weights)
            h_j.train()
            # 3. Append h_j to the self.classifiers attribute
            self.classifiers.append(h_j)
            # 4. Update the weights
            h_x = np.array([h_j.predict(scores[x, :]) for x in range(scores.shape[0])], np.float32) 
            eps_j = h_j.error   
            beta_j = eps_j / (1 - eps_j)
            e_i = np.float32(np.not_equal(h_x, self.labels)) * 2.0 - 1.0
            weights = np.multiply(weights, np.power(beta_j, 1.0 - e_i))
            # 5. Calculate alpha_j
            alpha_j = - np.log(beta_j)
            # Append it to the self.alphas
            self.alphas.append(alpha_j)



    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'
        for i in range(len(ii)):
            scores[i, :] = [hf.evaluate(ii[i]) for hf in self.haarFeatures]

        # Each col represents an image data
        predict_weights = np.empty([len(self.classifiers), len(ii)], np.float32)

        for i in range(len(self.classifiers)):
            # use each weak classifier to predict and store in a row
            predict_weights[i, :] = self.alphas[i] * np.array([self.classifiers[i].predict(scores[x_temp, :]) for x_temp in range(scores.shape[0])], np.float32)

        y_predict = np.sum(predict_weights, axis=0, dtype=np.float32)
        threshold_alphas = 0.5 * np.sum(np.array(self.alphas), dtype=np.float32)
        y_predict = np.float32(y_predict >= threshold_alphas) * 2.0 - 1.0

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        return y_predict.tolist()


    def faceDetection(self, image, filedir, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_final = image.copy()
        for row in range(11, img_gray.shape[0] - 12):
            for col in range(11, img_gray.shape[1] - 12):
                patch = img_gray[row - 11:row + 12, col - 11:col + 12]
                if self.predict([patch])[0] == 1.0:
                    cv2.rectangle(image_final, (col - 11, row - 11), (col + 12, row + 12), (0, 255, 0), 1)

        cv2.imwrite(os.path.join(filedir, filename), image_final)
