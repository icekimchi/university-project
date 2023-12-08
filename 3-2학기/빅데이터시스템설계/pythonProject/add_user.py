from flask import Flask, render_template, request
import pymysql
from faker import Faker
import random
import pandas as pd

app = Flask(__name__)

db = pymysql.connect(host="localhost", user="root", passwd="0000", db="mydb", charset="utf8")

#----------------------- 더미데이터 생성 -------------------------
fake = Faker('ko_KR')
Faker.seed(2)

repeat_count = 10000
#------------------------ user 테이블 ------------------------
# 커서 생성
cursor = db.cursor()

# 데이터 삽입 쿼리
insert_query = """
    INSERT INTO user (u_id, u_pw, u_name, u_email, u_phone)
    VALUES (%s, %s, %s, %s, %s)
"""

data = []
for _ in range(repeat_count):
    name = fake.name()
    phone = ('010-'+str(random.randint(1, 9999)).zfill(4)
        +'-'+str(random.randint(1, 9999)).zfill(4))
    id 	= fake.user_name()
    pw = fake.password()
    email = fake.unique.free_email()

    # 데이터 삽입
    cursor.execute(insert_query, (id, pw, name, email, phone))

# 변경사항 커밋
db.commit()

# 연결 종료
cursor.close()
db.close()


