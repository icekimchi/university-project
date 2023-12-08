import pymysql
import random

# MySQL 연결
db = pymysql.connect(host="localhost", user="root", passwd="0000", db="mydb", charset="utf8")

# 커서 생성
cursor = db.cursor()

# Get a list of all place IDs from the place table
place_ids_query = "SELECT p_id FROM place"
cursor.execute(place_ids_query)
place_ids = [row[0] for row in cursor.fetchall()]

# Iterate over each place ID
for place_id in place_ids:
    # Generate a random number of bicycle stands between 3 and 10
    num_bicycle_stands = random.randint(3, 10)

    # Insert records for each bicycle stand
    for stand_number in range(1, num_bicycle_stands + 1):
        insert_query = """
            INSERT INTO bicycle (place_p_id)
            VALUES (%s)
        """
        cursor.execute(insert_query, (place_id,))
        db.commit()

# 연결 종료
cursor.close()
db.close()
