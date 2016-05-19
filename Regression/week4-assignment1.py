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

sales = pd.read_csv('../datasets/kc_house_data.csv', dtype = dtype_dict)
sales = sales.sort_values(by = ['sqft_living','price'])

def polynomial_dataframe(feature, degree):

    poly_dataframe = pd.DataFrame()

    poly_dataframe['power_1'] = feature

    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = poly_dataframe.power_1.apply(lambda x : math.pow(x, power))


    return poly_dataframe

l2_small_penalty = 1.5e-5
degrees = 15

poly15_data = polynomial_dataframe(sales['sqft_living'], degrees)
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(poly15_data, sales['price'])

print("The coef are {}".format(model.coef_))

sales_set1 = pd.read_csv('../datasets/regression-week3/wk3_kc_house_set_1_data.csv', dtype = dtype_dict)
sales_set2 = pd.read_csv('../datasets/regression-week3/wk3_kc_house_set_2_data.csv', dtype = dtype_dict)
sales_set3 = pd.read_csv('../datasets/regression-week3/wk3_kc_house_set_3_data.csv', dtype = dtype_dict)
sales_set4 = pd.read_csv('../datasets/regression-week3/wk3_kc_house_set_4_data.csv', dtype = dtype_dict)


poly_data1 = polynomial_dataframe(sales_set1['sqft_living'], degrees)
poly_data2 = polynomial_dataframe(sales_set2['sqft_living'], degrees)
poly_data3 = polynomial_dataframe(sales_set3['sqft_living'], degrees)
poly_data4 = polynomial_dataframe(sales_set4['sqft_living'], degrees)
l2_small_penalty = 1e-9

model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
print("First set coef are {}".format(model.fit(poly_data1, sales_set1['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data2, sales_set2['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data3, sales_set3['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data4, sales_set4['price']).coef_[0]))
print('The intercept is {}\n'.format(model.intercept_))

##############################
l2_large_penalty = 1.23e2
model = linear_model.Ridge(alpha=l2_large_penalty, normalize=True)

print("First set coef are {}".format(model.fit(poly_data1, sales_set1['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data2, sales_set2['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data3, sales_set3['price']).coef_[0]))
print("First set coef are {}".format(model.fit(poly_data4, sales_set4['price']).coef_[0]))
print('The intercept is {}\n'.format(model.intercept_))

##########################################################################################
##########################################################################################

train_valid_shuffled = pd.read_csv('../datasets/regression-week3/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('../datasets/regression-week3/wk3_kc_house_test_data.csv', dtype=dtype_dict)

n = len(train_valid_shuffled)
k = 10



def k_fold_cross_validation(k, l2_penalty, data, output):

    rss_sum = 0
    for i in range(k):
        start = math.floor((n * i) / k)
        end = math.floor((n * (i + 1)) / k - 1)
        # print(i, (start, end))

        valid_set = data[start:end+1]
        training_set = data[0:start].append(data[end+1:n])
        model = linear_model.Ridge(alpha=l2_penalty, normalize=True)

        model.fit(training_set, output[0:start].append(output[end+1:n]))

        rss_sum += sum((output[start:end+1] - model.predict(valid_set)) ** 2)

    return (rss_sum/k)

poly_train_valid_data = polynomial_dataframe(train_valid_shuffled['sqft_living'], 15)

for l2 in np.logspace(3, 9, num=13):
    # print(l2)
    min_rss = float("inf")
    rss = k_fold_cross_validation(k, l2, poly_train_valid_data, train_valid_shuffled['price'])
    if min_rss > rss:
        min_rss = rss
        min_l2 = l2

#


model = linear_model.Ridge(alpha=1e3, normalize=True)
model.fit(poly_train_valid_data,train_valid_shuffled['price'])
poly_test = polynomial_dataframe(test['sqft_living'], 15)
rss = sum((test['price'] - model.predict(poly_test)) ** 2)

print("rss is {:e}".format(rss))


