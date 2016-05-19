import pandas as pd
from math import log, sqrt
from sklearn import linear_model
import numpy as np
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('../datasets/kc_house_data.csv', dtype=dtype_dict)

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']

model_all = linear_model.Lasso(alpha=5e2, normalize=True)
model_all.fit(sales[all_features], sales['price'])
print("non zero features are ", model_all.coef_)

testing = pd.read_csv('../datasets/regression-week3/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('../datasets/regression-week3/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('../datasets/regression-week3/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

for l1 in np.logspace(1, 7, num=13):
    model = linear_model.Lasso(alpha=l1, normalize=True)
    model.fit(training[all_features], training['price'])
    rss = sum((model.predict(validation[all_features]) - validation['price']) ** 2)
    print("The rss for {} is {:e}".format(l1, rss))


model = linear_model.Lasso(alpha=10, normalize=True)
model.fit(training[all_features], training['price'])
print(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))

max_nonzeros = 7
l1_penalty_min = 1
l1_penalty_max = 100000
for l1_penalty in np.logspace(1, 4, num=20):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])
    num_nonzero = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

    if (num_nonzero > max_nonzeros) & (l1_penalty > l1_penalty_min):
        print(l1_penalty)
        l1_penalty_min = l1_penalty

    if (num_nonzero < max_nonzeros) & (l1_penalty < l1_penalty_max):
        print(l1_penalty)
        l1_penalty_max = l1_penalty

print(l1_penalty_min, l1_penalty_max)

li = {}
for l1_penalty in np.linspace(l1_penalty_min, l1_penalty_max, 20):
    model = linear_model.Lasso(alpha=l1_penalty, normalize=True)
    model.fit(training[all_features], training['price'])

    rss = sum((model.predict(validation[all_features]) - validation['price']) ** 2)
    num_nonzero = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    print("The rss for {} is {:e}, non zero is {}".format(l1_penalty, rss, num_nonzero))

    if num_nonzero == max_nonzeros:
        li[l1_penalty] = rss


min_l1 = min(li.items(), key=lambda x: x[1])[0]
#2.
# min_l1 = [ item[0] for item in li if item[1] == min(li.values())]
print(min_l1)

model = linear_model.Lasso(alpha=min_l1, normalize=True)
model.fit(training[all_features], training['price'])
print(model.coef_)