import pandas as pd
import numpy as np
from sklearn import linear_model
import math
dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str, 'long': float,
              'sqft_lot15': float, 'sqft_living': float, 'floors': str, 'condition': int, 'lat': float, 'date': str,
              'sqft_basement': int, 'yr_built': int, 'id': str, 'sqft_lot': int, 'view': int}

train_data = pd.read_csv(
    "../datasets/kc_house_train_data.csv", dtype=dtype_dict)
test_data = pd.read_csv(
    "../datasets/kc_house_test_data.csv", dtype=dtype_dict)

train_data["bedrooms_squared"] = train_data.bedrooms * train_data.bedrooms
train_data["bed_bath_rooms"] = train_data.bedrooms * train_data.bathrooms
train_data["log_sqft_living"] = np.log(train_data.sqft_living)
train_data["lat_plus_long"] = train_data.lat + train_data.long

test_data["bedrooms_squared"] = test_data.bedrooms * test_data.bedrooms
test_data["bed_bath_rooms"] = test_data.bedrooms * test_data.bathrooms
test_data["log_sqft_living"] = np.log(test_data.sqft_living)
test_data["lat_plus_long"] = test_data.lat + test_data.long

# print(test_data["bedrooms_squared"].mean)
print("The befrooms_squared AVG is {:.2f}".format(test_data["bedrooms_squared"].mean()))
print("The bed_bath_rooms AVG is {:.2f}".format(test_data["bed_bath_rooms"].mean()))
print("The log_sqft_living AVG is {:.2f}".format(test_data["log_sqft_living"].mean()))
print("The lat_plus_long AVG is {:.2f}".format(test_data["lat_plus_long"].mean()))

###############

clf1 = linear_model.LinearRegression()
clf2 = linear_model.LinearRegression()
clf3 = linear_model.LinearRegression()

model1_feature = ["sqft_living", "bedrooms", "bathrooms", "lat", "long"]
model2_feature = ["sqft_living", "bedrooms", "bathrooms", "lat", "long", "bed_bath_rooms"]
model3_feature = ["sqft_living", "bedrooms", "bathrooms", "lat",
                                 "long", "bed_bath_rooms", "bedrooms_squared", "log_sqft_living"]

model1_data = train_data.loc[:, model1_feature]
model1_data_double = model1_data.copy()
model1_data_double["sqft_living"] = model1_data_double.sqft_living * 2
test1_data = test_data.loc[:, model1_feature]
model1 = clf1.fit(model1_data, y=train_data.price)

model2_data = train_data.loc[:, model2_feature]
test2_data = test_data.loc[:, model2_feature]
model2 = clf2.fit(model2_data, y=train_data.price)

model3_data = train_data.loc[:, model3_feature]
test3_data = test_data.loc[:, model3_feature]
model3 = clf3.fit(model3_data, y=train_data.price)

print("The coeff for model1 is {}".format(model1.coef_))
print("The coeff for model2 is {}".format(model2.coef_))

# print(np.dot(model1.coef_, [1, 2, 3, 4, 5]))


def calrss(input_features, output, clf):

    # hw = np.dot(input_features.as_matrix(),coef.T)
    # rss_sum = (output.as_matrix()-hw) * (output.as_matrix()-hw)
    rss_sum =((output - clf.predict(input_features)) ** 2)

    return sum(rss_sum)


result = train_data["price"]

model1_rss = calrss(model1_data, result, model1)
model2_rss = calrss(model2_data, result, model2)
model3_rss = calrss(model3_data, result, model3)

print("Model1 RSS is {}, Model2 RSS is {}, Modelis {}".format(model1_rss,model2_rss,model3_rss))

##############################
test_res = test_data["price"]
test1_rss = calrss(test1_data, test_res, model1)
test2_rss = calrss(test2_data, test_res, model2)
test3_rss = calrss(test3_data, test_res, model3)
print("Model1 RSS is {:e}, Model2 RSS is {}, Model3 RSS is {}".format(test1_rss, test2_rss, test3_rss))

# If we double the value of one feature, what would happen to the coefficients of the rest features?
model_double = linear_model.LinearRegression().fit(model1_data_double, result)
print("The double model parameter is {} ".format(model_double.coef_))