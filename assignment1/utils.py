import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    tp, tn, fp, fn = 0, 0, 0, 0
    for real_label, predicted_label in zip(real_labels, predicted_labels):
        if real_label == 1 and predicted_label == 1:
            tp += 1
        elif real_label == 0 and predicted_label == 1:
            fp += 1
        elif real_label == 1 and predicted_label == 0:
            fn += 1
        else:
            tn += 1 
        
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)

    if precision + recall == 0:
        return 0
    else:
        # f1 = 2 * (precision * recall) / (precision + recall)
        return 2.0 * (precision * recall) / (precision + recall)

class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = np.array(point1)
        y = np.array(point2)

        # np.fabs computes absolute values element-wise
        dist = np.nansum(np.fabs(x - y) / (np.fabs(x) + np.fabs(y))) 
        return dist

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = np.asarray(point1)
        y = np.asarray(point2)

        p = 3
        dist = np.power(np.sum(np.power(np.fabs(x - y), p)), 1. / p)
        return dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = np.asarray(point1)
        y = np.asarray(point2)

        dist = np.sqrt(np.sum(np.dot((x - y), (x - y))))
        return dist

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        x = np.asarray(point1)
        y = np.asarray(point2)

        dist = np.inner(x, y)
        return dist

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x = np.asarray(point1)
        y = np.asarray(point2)

        dist = 1.0 - (np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))) if (np.linalg.norm(x) * np.linalg.norm(y)) != 0 else 0
        return dist

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        x = np.asarray(point1)
        y = np.asarray(point2)

        sub = np.subtract(x,y)
        return -np.exp(-1/2*np.inner(sub,sub))


        # dist = -np.exp(-1./2*(np.inner(x - y, x-y)))
        # return dist


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = 0                    
        self.best_distance_function = None # string
        self.best_model = None
         
        k_values = range(1, 30, 2)
        score = 0
        best_score = -1
        i = 0
        distance_metric = list(distance_funcs.keys())
        best_metrics = {}

        for k in k_values:
            for dist_func_name, dist_func in distance_funcs.items():
                score = 0

                # train the model and compute f1 score
                model = KNN(k, dist_func)
                model.train(x_train, y_train)
                predicted_labels = model.predict(x_val)
                score = f1_score(y_val, predicted_labels)

                dist_func_index = distance_metric.index(dist_func_name)

                if score >= best_score:
                    best_score = score
                    best_metrics[i] = ({
                            'score': score,
                            'distance_fn': dist_func_index,
                            'k': k
                            })

                i += 1

        # get sorted knn result based on score, distance metrics and k 
        best_metrics = {k: v for k, v in sorted(best_metrics.items(), key=lambda item: (-item[1]['score'], item[1]['distance_fn'], item[1]['k']))}
        best_metrics = list(best_metrics.values())[0]  # take the first one as the best metrics 
        self.best_k = best_metrics['k']
        self.best_distance_function = distance_metric[best_metrics['distance_fn']]
        
        best_distance_fn = distance_funcs[self.best_distance_function]
        self.best_model = KNN(self.best_k, best_distance_fn)
        self.best_model.train(x_train, y_train)
                    
    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = 0
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        k_values = range(1, 30, 2)
        score = 0
        best_score = -1
        i = 0
        best_metrics = {}
        distance_metric = list(distance_funcs.keys())
        scalar_mtd = list(scaling_classes.keys())

        for scaling_name, scaling_class in scaling_classes.items(): 
            # transform train and val data
            scaling_func = scaling_class()
            x_train_scaled = scaling_func(x_train)
            x_val_scaled = scaling_func(x_val)

            for k in k_values:
                for dist_func_name, dist_func in distance_funcs.items():

                    # train the model with transformed data
                    model = KNN(k, dist_func)
                    model.train(x_train_scaled, y_train)
                    predicted_labels = model.predict(x_val_scaled)
                    score = f1_score(y_val, predicted_labels)

                    dist_func_index = distance_metric.index(dist_func_name)
                    scalar_index = scalar_mtd.index(scaling_name)

                    if score >= best_score:
                        best_score = score
                        best_metrics[i] = ({
                                'score': score,
                                'distance_fn': dist_func_index,
                                'scalar': scalar_index,
                                'k': k
                                })
                    i += 1

        best_metrics = {k: v for k, v in sorted(best_metrics.items(), key=lambda item: (-item[1]['score'], item[1]['scalar'], item[1]['distance_fn'], item[1]['k']))}
        best_metrics = list(best_metrics.values())[0] # take the first one as the best metrics 
        self.best_k = best_metrics['k']
        self.best_scaler = scalar_mtd[best_metrics['scalar']]
        self.best_distance_function = distance_metric[best_metrics['distance_fn']]
        
        best_distance_fn = distance_funcs[self.best_distance_function]
        best_scalar = scaling_classes[self.best_scaler]()
        self.best_model = KNN(self.best_k, best_distance_fn)
        self.best_model.train(best_scalar(x_train), y_train)

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norms = []
        for i in range(len(features)):
            norm = []
            scaling_w = np.sqrt(np.inner(features[i], features[i]))
            for j in range(len(features[i])):
                if scaling_w != 0:
                    norm.append(features[i][j]/scaling_w)
                else:
                    norm.append(0)
            norms.append(norm)
        return norms

class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.initialized = False
        self.data_max = []
        self.data_min = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        if not self.initialized:
            self.initialized = True
            data = np.reshape(features, (len(features),len(features[0])))
            self.data_max.extend(np.amax(data, axis=0))
            self.data_min.extend(np.amin(data, axis=0))
        
        # scale according to max and min of each column
        minimax = features.copy()
        for j in range(len(features[0])):
            for i in range(len(features)):
                denom = self.data_max[j] - self.data_min[j]
                if denom != 0:
                    minimax[i][j] = (minimax[i][j] - self.data_min[j])/denom
                else:
                    minimax[i][j] = 0.0     
        return minimax
