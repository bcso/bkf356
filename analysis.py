import os
import pymysql.cursors

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
        sql = "select * from `review` where `user_id`=%s"
        cursor.execute(sql, [user_id])
        reviews = cursor.fetchall()
finally:
    connection.close()

print reviews[0].keys()