import pandas as pd
import math
from sklearn.cross_validation import train_test_split

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

train_data = pd.read_csv(
    "../datasets/kc_house_train_data.csv", dtype=dtype_dict)
test_data = pd.read_csv(
    "../datasets/kc_house_test_data.csv", dtype=dtype_dict)
# dataset = pd.read_csv(
    # "../datasets/kc_house_data.csv", dtype=dtype_dict)

# train_data, test_data = train_test_split(dataset, test_size=0.2)

# print(dataset.dtypes)


def simple_linear_regression(input_features, output):
    output_sum = output.sum()
    input_sum = input_features.sum()
    input_output_product_sum = (input_features * output).sum()
    input_square_sum = pow(input_features, 2).sum()
    length = input_features.size


    # print(": ", output_sum / length, input_sum/ length, input_output_product_sum/ length, input_square_sum/ length)
    numerator = input_output_product_sum - \
        (1 / length) * (input_sum * output_sum)
    denominator = input_square_sum - (1 / length) * (input_sum * input_sum)

    slope = numerator / denominator

    # print("coap: ", input_square_sum, (input_sum * output_sum) / length)
    intercept = output_sum / length - (slope * (input_sum / length))
    print(intercept, slope)
    return(intercept, slope)


def get_regression_predictions(input_features, intercept, slope):

    return input_features * slope + intercept


def get_residual_sum_squares(input_features, output, intercept, slope):
    predict_outputs = get_regression_predictions(
        input_features, intercept, slope)

    # print(predict_outputs)
    RSS = pow((output - predict_outputs), 2).sum()

    # print("RSS: ", RSS)

    return RSS


def inverse_regression_predictions(output, intercept, slope):

    return (output - intercept) / slope

squarfeet_intercept, squarfeet_slope = simple_linear_regression(
    train_data.sqft_living, train_data.price)

quiz1 = get_regression_predictions(
    2650.0, squarfeet_intercept, squarfeet_slope)
print("quiz1: ", quiz1)

quiz2 = get_residual_sum_squares(
    train_data.sqft_living, train_data.price, squarfeet_intercept, squarfeet_slope)
print("quiz2: ", quiz2)

quiz3 = inverse_regression_predictions(
    800000.0, squarfeet_intercept, squarfeet_slope)
print("quiz3: ", quiz3)

bedroom_intercept, bedroom_slope = simple_linear_regression(
    train_data.bedrooms, train_data.price)

bedrooms_test_RSS = get_residual_sum_squares(
    test_data.bedrooms, test_data.price, bedroom_intercept, bedroom_slope)

squarfeet_test_RSS = get_residual_sum_squares(
    test_data.sqft_living, test_data.price, squarfeet_intercept, squarfeet_slope)


print("bedrooms_RSS: ", bedrooms_test_RSS)
print("squarfeet_RSS: ", squarfeet_test_RSS)
