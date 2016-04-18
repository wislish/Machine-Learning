import numpy as np
import math
import pandas as pd

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


def feature_derivative(errors, feature):

    derivative = 2 * np.dot(errors, feature)

    return derivative

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)

    while not converged:

        gradient_sum_squares = 0
        for i in range(len(weights)):

            errors = (output - predict_outcome(feature_matrix, weights))

            derivative = feature_derivative(errors, feature_matrix[:, i])
            weights[i] += step_size * derivative

            gradient_sum_squares += derivative ** 2

        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return (weights)


def calrss(input_features, output, weights):

    # hw = np.dot(input_features.as_matrix(),coef.T)
    # rss_sum = (output.as_matrix()-hw) * (output.as_matrix()-hw)
    rss_sum =((output - predict_outcome(input_features, weights)) ** 2)

    return sum(rss_sum)


simple_features = ["sqft_living"]
my_output = 'price'
(simple_features_matrix, output)  = get_numpy_data(train_data, simple_features,my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_features_matrix,output,initial_weights,step_size,tolerance)

print("the weight of simple feature are {}".format(simple_weights))

test_simple_feature_matrix, test_output = get_numpy_data(test_data, simple_features, my_output)
print(test_simple_feature_matrix[0])
predict_price = predict_outcome(test_simple_feature_matrix, simple_weights)
print("The predicted price for model 1 is {}".format(predict_price[0]))


simple_rss = calrss(simple_features_matrix, output, simple_weights)
print("The simple model's rss is {:e}".format(simple_rss))

###############################

model_features = ["sqft_living","sqft_living15"]
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9


multi_model_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
test_feature_matrix, test_output = get_numpy_data(test_data, model_features, my_output)
print(test_feature_matrix[0])
predict_price = predict_outcome(test_feature_matrix, multi_model_weights)
print("The predict price is {}".format(predict_price[0]))

multi_rss = calrss(feature_matrix, output, multi_model_weights)
print("The model rss is {:e}".format(multi_rss))

print(test_output[0])
print(test_data.iloc[0,:])

