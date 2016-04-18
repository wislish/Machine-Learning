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


def polynomial_dataframe(feature, degree):

    poly_dataframe = pd.DataFrame()

    poly_dataframe['power_1'] = feature

    if degree > 1:
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_dataframe[name] = poly_dataframe.power_1.apply(lambda x : math.pow(x, power))


    return poly_dataframe

def plotandlinear(sales, degrees):

    sales = sales.sort_values(by = ['sqft_living','price'])
    poly_data = polynomial_dataframe(sales['sqft_living'], degrees)
    col_names = poly_data.columns
    poly_data['price'] = sales['price']
    clf = linear_model.LinearRegression()
    print(poly_data[col_names])
    model = clf.fit(pd.DataFrame(poly_data[col_names]), y = poly_data["price"])
    print(model.coef_)
    print(poly_data.columns)


sales = pd.read_csv('../datasets/kc_house_data.csv', dtype = dtype_dict)
sales = sales.sort_values(by = ['sqft_living','price'])
poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']
# print(poly1_data.iloc[:,1])

clf1 = linear_model.LinearRegression()
model1 = clf1.fit(pd.DataFrame(poly1_data["power_1"]), y = poly1_data["price"])
model1_coef = model1.coef_
model1_intercept = model1.intercept_

# poly1_plot = sns.jointplot(x = "sqft_living", y = "price", data = sales)
# sns.lmplot(x = "sqft_living", y = "price", data = sales)

sales_set1 = pd.read_csv('../datasets/week3/wk3_kc_house_set_1_data.csv', dtype = dtype_dict)
sales_set2 = pd.read_csv('../datasets/week3/wk3_kc_house_set_2_data.csv', dtype = dtype_dict)
sales_set3 = pd.read_csv('../datasets/week3/wk3_kc_house_set_3_data.csv', dtype = dtype_dict)
sales_set4 = pd.read_csv('../datasets/week3/wk3_kc_house_set_4_data.csv', dtype = dtype_dict)

plotandlinear(sales_set1, 15)
