import os
import pymysql.cursors
from pprint import pprint
from sklearn import datasets, linear_model
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import accuracy_score

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             # Set your password by typing this into your shell: export MYSQL_PASS='your_password'
                             password=os.environ.get("MYSQL_PASS", ''),
                             db='yelp_db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        sql = "select r_date, r_count, r_useful, r_funny, r_cool, r_stars, b.id b_id, b.neighborhood b_hood, b.name b_name, b.address b_address, b.city b_city, b.state b_state, b.postal_code b_pc, b.latitude b_lat, b.longitude b_longi, b.stars b_stars, b.review_count b_review_count from business b join (select business_id, sum(useful) r_useful, sum(funny) r_funny, sum(cool) r_cool, sum(stars) r_stars, count(*) r_count, min(DATEDIFF(date, '1970/01/01')) r_date from review group by business_id) r on b.id = r.business_id where b.latitude IS NOT NULL"
        cursor.execute(sql)
        reviews = cursor.fetchall()
finally:
    connection.close()

y = []
X = []
for review in reviews:
    row = {
        'b_stars': review['b_stars'],
        'b_lat': review['b_lat'],
        'b_longi': review['b_longi'],
        'r_stars': review['r_stars'],
        'r_cool': review['r_cool'],
        'r_funny': review['r_funny'],
        'r_useful': review['r_useful'],
        'r_date': int(review['r_date']),
    }
    X.append(row)
    y.append(review['b_review_count'])

vec = DictVectorizer()
vectorized_features = vec.fit_transform(X).toarray()
feature_names = vec.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, y, test_size=0.5, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
# Rounding to the nearest integer
y_pred = regr.predict(X_test)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Explained Variance score: %.2f' % explained_variance_score(y_test, y_pred))
print('R2 score: %.2f' % r2_score(y_test, y_pred))

print regr.coef_