import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

sales = pd.read_csv('../datasets/kc_house_data.csv', dtype=dtype_dict)

def get_numpy_data(data_frame, features, output):

    data_frame['intercept'] = 1
    features = ['intercept'] + features

    features_matrix = data_frame[features].as_matrix()

    output_array = data_frame[output].values
    return (features_matrix, output_array)

def predict_outcome(feature_matrix, weights):

    return np.dot(feature_matrix, weights)

def normalize_features(features):

    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms

    return (normalized_features, norms)

def calrss(input_features, output, weights):

    # hw = np.dot(input_features.as_matrix(),coef.T)
    # rss_sum = (output.as_matrix()-hw) * (output.as_matrix()-hw)
    rss_sum =((output - predict_outcome(input_features, weights)) ** 2)

    return sum(rss_sum)


features, output = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
normalized_features, norms = normalize_features(features)
initial_weights = [1., 4., 1.]

initial_prediction = predict_outcome(normalized_features, initial_weights)

for i in [1, 2]:
    ro = sum(normalized_features[:, i] * (output - initial_prediction + initial_weights[i] * normalized_features[:, i]))
    print(" {:e}".format(ro))

###############################

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):

    prediction = predict_outcome(feature_matrix, weights)

    ro_i = sum(feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0

    return new_weight_i

# should print 0.425558846691

print (lasso_coordinate_descent_step(1, np.array([[3. / math.sqrt(13), 1. / math.sqrt(10)],
                                           [2. / math.sqrt(13), 3. / math.sqrt(10)]]), np.array([1., 1.]),
                              np.array([1., 4.]), 0.1))
############################################################

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):

    max_coordiante_change = float('inf')
    weights = initial_weights

    while True:
        diff_weights = []
        for i in range(len(initial_weights)):
            new_weight = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            diff_weights.append(new_weight - weights[i])
            weights[i] = new_weight

        # print(new_weights)
        diff_weights = [ math.fabs(i) for i in diff_weights]
        # print(diff_weights)

        if max(diff_weights) < tolerance:
            return weights

initial_weights = [0., 0., 0.]
l1_penalty = 1e7
tolerance = 1.0

new_weights = lasso_cyclical_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print("new weights are ", new_weights)
rss = sum((predict_outcome(normalized_features, new_weights) - output) ** 2)
print("rss is ", rss)


training = pd.read_csv('../datasets/kc_house_train_data.csv', dtype=dtype_dict)
test= pd.read_csv('../datasets/kc_house_test_data.csv', dtype=dtype_dict)

features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']

test_matrix, test_output = get_numpy_data(test, features, 'price')
features_matrix, output = get_numpy_data(training, features, 'price')
normalized_features, norms = normalize_features(features_matrix)
initial_weights = np.zeros(len(features)+1)
l1_penalty = 1e7
tolerance =1

weights1e7 = lasso_cyclical_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print("weights1e7 is ", weights1e7)
normalized_weights1e7 = np.divide(np.array(weights1e7), np.array(norms))
print ("test for weights1e7", calrss(test_matrix, test_output, normalized_weights1e7))
print (normalized_weights1e7[3])


l1_penalty = 1e8
weights1e8 = lasso_cyclical_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print("weights1e8 is ", weights1e8)
normalized_weights1e8 = np.divide(np.array(weights1e8), np.array(norms))
print ("test for weights1e8", calrss(test_matrix, test_output, normalized_weights1e8))

l1_penalty = 1e4
tolerance = 5e5
weights1e4= lasso_cyclical_coordinate_descent(normalized_features, output, initial_weights, l1_penalty, tolerance)
print("weights1e4 is ", weights1e4)
normalized_weights1e4 = np.divide(np.array(weights1e4), np.array(norms))
print ("test for weights1e4", calrss(test_matrix, test_output, normalized_weights1e4))

