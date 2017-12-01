import os
import pymysql.cursors
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import graphviz
from numpy import shape
from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO  
# import pydot 

from sklearn.ensemble import RandomForestClassifier

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
        sql = "select c.category c_category, b.id b_id, b.neighborhood b_hood, b.name b_name, b.address b_address, b.city b_city, b.state b_state, b.postal_code b_pc, b.latitude b_lat, b.longitude b_longi, b.stars b_stars, b.review_count b_review_count, r.stars r_stars from business b join review r on r.business_id = b.id join category c on c.business_id = b.id where r.user_id = %s"
        cursor.execute(sql, [user_id])
        reviews = cursor.fetchall()
finally:
    connection.close()

y = []
X = []
for review in reviews:
    row = {
        'b_review_count': review['b_review_count'],
        'b_hood': review['b_hood'],
        'b_name': review['b_name'],
        'b_address': review['b_address'],
        'b_city': review['b_city'],
        'b_state': review['b_state'],
        'b_stars': review['b_stars'],
        # 'b_postal_code': review['b_pc'],
        'c_category': review['c_category'],
    }
    X.append(row)
    y.append(review['r_stars'])

vec = DictVectorizer()
vectorized_features = vec.fit_transform(X).toarray()
feature_names = vec.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(vectorized_features, y, test_size=0.5, random_state=42)
dt = DecisionTreeClassifier(min_samples_split=100, random_state=99)
dt = dt.fit(X_train, y_train)

dot_data = export_graphviz(dt, out_file='test.dot', 
                         feature_names=feature_names,  
                         class_names=['1','2','3','4','5'],
                         filled=True, rounded=True,  
                         special_characters=True) 

y_pred = dt.predict(X_test)
print('Decision Tree accuracy: %.2f' % (accuracy_score(y_test,y_pred)*100))


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Random Forest accuracy: %.2f' % (accuracy_score(y_test,y_pred)*100))