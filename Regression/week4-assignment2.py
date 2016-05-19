import pandas as pd
import numpy as np
import math
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

train_data = pd.read_csv(
    "../datasets/kc_house_train_data.csv", dtype=dtype_dict)
test_data = pd.read_csv(
    "../datasets/kc_house_test_data.csv", dtype=dtype_dict)

def get_numpy_data(data_frame, features, output):

    data_frame['intercept'] = 1
    features = ['intercept'] + features

    features_matrix = data_frame[features].as_matrix()

    output_array = data_frame[output].values
    return (features_matrix, output_array)


def predict_outcome(feature_matrix, weights):

    return np.dot(feature_matrix, weights)


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):

    rss_derivative = 2 * np.dot(errors, feature)
    if feature_is_constant:

        return  rss_derivative

    else:
        l2_derivative = 2 * l2_penalty * weight
        return (rss_derivative + l2_derivative)


def ridge_regression_gradient_discent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations = 100):

    weights = np.array(initial_weights)
    iterations = 0
    while iterations < max_iterations:

        for i in range(len(weights)):
            if i == 0:
                feature_is_constant = True
            else:
                feature_is_constant = False
            errors = predict_outcome(feature_matrix, weights) - output
            weights[i] -= step_size * feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, feature_is_constant)
        iterations = iterations + 1
        print(iterations)

    return weights

simple_features = ['sqft_living']
my_output = 'price'

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

step_size = 1e-12
max_iterations = 1000
initial_weights = [0., 0.]
high_penalty = 1e11

simple_weights_0_penalty = ridge_regression_gradient_discent(simple_feature_matrix, output, initial_weights, step_size,
                                                             0.0,
                                                             max_iterations)
simple_weights_high_penalty = ridge_regression_gradient_discent(simple_feature_matrix, output, initial_weights, step_size,
                                                                high_penalty,
                                                                max_iterations)

# plt.plot(simple_feature_matrix,output,'k.',
#         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
#         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')


