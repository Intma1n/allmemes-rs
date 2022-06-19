import psycopg2
import os
from pprint import pprint
from dotenv import load_dotenv
from random import randint

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

db_name = os.environ.get('DW_DATABASE_NAME')
db_user = os.environ.get('DW_DATABASE_USER')
db_password = os.environ.get('DW_DATABASE_PASSWORD')
db_host = os.environ.get('DW_DATABASE_HOST')
db_port = os.environ.get('DW_DATABASE_PORT')


def get_dw_connection():
    """
    Connection to dw database
    :return:
    """
    my_conn = psycopg2.connect(dbname=db_name,
                               host=db_host,
                               port=db_port,
                               user=db_user,
                               password=db_password)
    return my_conn


dw_conn = get_dw_connection()


def select_data_from_db(request):
    with dw_conn.cursor() as curs:
        sql = request
        curs.execute(sql)
        query_results = curs.fetchall()
    return query_results


# def update_data_in_db():
#     with dw_conn.cursor() as curs:
#         sql = f"update pictures set likes[1] = {randint(1, 14)} where index = {randint(1, 75)};"
#         curs.execute(sql)
#         query_results = curs.fetchall()
#     return query_results


# get all liked pics
print(select_data_from_db(f"select * from pictures where likes[1] IS NOT NULL;"))
# get all users
print(select_data_from_db(f"select * from users"))
# build matrix users/liked pics
print("-------------------------------------------------------------------------------------")
print(select_data_from_db(f"select pics.index, pics.picture, users.id, users.username "
                          f"from pictures pics"
                          f" inner join users"
                          f" on users.id = any (pics.likes);"))
