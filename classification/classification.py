import os
import pymysql.cursors
from pprint import pprint
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import graphviz

from sklearn.externals.six import StringIO  
import pydot 

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
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt = dt.fit(vectorized_features, y)

dot_data = export_graphviz(dt, out_file='test.dot', 
                         feature_names=feature_names,  
                         class_names=df1['r_stars'],
                         filled=True, rounded=True,  
                         special_characters=True) 