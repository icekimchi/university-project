import pymysql
from faker import Faker
import random
from datetime import datetime, timedelta

# MySQL 연결
db = pymysql.connect(host="localhost", user="root", passwd="0000", db="mydb", charset="utf8")

# 커서 생성
cursor = db.cursor()

# 더미데이터 생성을 위한 Faker 설정
fake = Faker('ko_KR')
Faker.seed(2)

# 사용자 번호 (u_num) 리스트 생성
user_nums_query = "SELECT u_num FROM user"
cursor.execute(user_nums_query)
user_nums = [row[0] for row in cursor.fetchall()]

# 거치소 정보를 가져오기
place_ids_query = "SELECT p_id FROM place"
cursor.execute(place_ids_query)
place_ids = [row[0] for row in cursor.fetchall()]

# 자전거 정보를 가져오기
bicycle_ids_query = "SELECT b_id FROM bicycle"
cursor.execute(bicycle_ids_query)
bicycle_ids = [row[0] for row in cursor.fetchall()]

# 대여 및 반납 기록 생성
for _ in range(1000):  # 예시로 1000개의 대여 기록 생성
    # 무작위로 사용자 선택
    u_num = random.choice(user_nums)

    # 무작위로 거치소 선택
    s_place = random.choice(place_ids)
    e_place = random.choice(place_ids)

    # 무작위로 자전거 선택
    bicycle_id = random.choice(bicycle_ids)

    # 가상의 대여 및 반납 시간 생성 (현재 시간을 기준으로 무작위로 생성)
    s_time = fake.date_time_this_decade()
    e_time = s_time + timedelta(minutes=random.randint(10, 120))  # 10분부터 2시간 사이의 대여 기간

    # 대여 비용은 무작위로 설정
    h_price = random.randint(1000, 5000)

    # 대여 기록 삽입
    insert_query = """
        INSERT INTO history (h_price, s_time, e_time, bicycle_b_id, s_place, e_place, u_num)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (h_price, s_time, e_time, bicycle_id, s_place, e_place, u_num))
    db.commit()

# 연결 종료
cursor.close()
db.close()
