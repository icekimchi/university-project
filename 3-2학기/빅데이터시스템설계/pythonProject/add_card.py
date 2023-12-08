'''
----------------------- 모듈 다운로드 --------------------
'''
import pymysql
from faker import Faker
import random

# MySQL 연결
db = pymysql.connect(host="localhost", user="root", passwd="0000", db="mydb", charset="utf8")

# 더미데이터 생성을 위한 Faker 설정
fake = Faker('ko_KR')
Faker.seed(2)

# 데이터 삽입 쿼리
insert_card_query = """
    INSERT INTO card (c_name, c_num, c_CVC, c_valid, u_num)
    VALUES (%s, %s, %s, %s, %s)
"""

# 커서 생성
cursor = db.cursor()

# 더미 데이터 생성 및 삽입
for i in range(1, 10001):
    for _ in range(random.randrange(1, 3)):
        # 랜덤 은행 이름 생성
        korean_banks = ['국민은행', '우리은행', '신한은행', '하나은행', '기업은행', '농협은행']
        bank_name = fake.random_element(elements=korean_banks)

        # 랜덤한 카드 번호 생성 (16자리)
        card_number = fake.credit_card_number(card_type='mastercard')

        # 랜덤한 CVC 번호 생성 (3자리)
        cvc_number = fake.credit_card_security_code(card_type='mastercard')

        # 랜덤한 유효 기간 생성 (MM/YY 형식)
        expiration_date = fake.credit_card_expire()

        # 데이터 삽입
        cursor.execute(insert_card_query, (bank_name, card_number, cvc_number, expiration_date, i))

# 변경사항 커밋
db.commit()

# 연결 종료
cursor.close()
db.close()