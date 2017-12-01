import os
import pymysql.cursors
from pprint import pprint
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             # Set your password by typing this into your shell: export MYSQL_PASS='your_password'
                             password=os.environ.get("MYSQL_PASS", ''),
                             db='yelp_db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

user_id = 'ELcQDlf69kb-ihJfxZyL0A'

try:
    with connection.cursor() as cursor:
        sql = "select b.id b_id, b.neighborhood b_hood, b.name b_name, b.address b_address, b.city b_city, b.state b_state, b.postal_code b_pc, b.latitude b_lat, b.longitude b_longi, b.stars b_stars, b.review_count b_review_count, r.stars r_stars from business b join review r on r.business_id = b.id where r.user_id = %s"
        cursor.execute(sql, [user_id])
        reviews = cursor.fetchall()
finally:
    connection.close()

vec = DictVectorizer()
vectorized_features = vec.fit_transform(reviews).toarray()
feature_names = vec.get_feature_names()

df1 = pd.DataFrame(reviews)

y = df1['r_stars']
X_train, X_test, y_train, y_test = train_test_split(vectorized_features, y, test_size=0.5, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))