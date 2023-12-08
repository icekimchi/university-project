from flask import Flask, render_template, request
import pymysql

app = Flask(__name__)

# MySQL 연결
db = pymysql.connect(host="localhost", user="root", passwd="0000", db="mydb", charset="utf8")

# 행정구역 리스트
regions = [
    '강남구', '강동구', '강북구', '강서구', '관악구',
    '광진구', '구로구', '금천구', '노원구', '도봉구',
    '동대문구', '동작구', '마포구', '서대문구', '서초구',
    '성동구', '성북구', '송파구', '양천구', '영등포구',
    '용산구', '은평구', '종로구', '중구', '중랑구'
]

@app.route('/', methods=['GET'])
def index():
    selected_region = request.args.get('region', None)

    if selected_region:
        cursor = db.cursor()
        query = f"SELECT * FROM place WHERE add_1 LIKE '%{selected_region}%'"
        cursor.execute(query)
        result = cursor.fetchall()
    else:
        result = None

    return render_template('index.html', regions=regions, selected_region=selected_region, result=result)



@app.route('/user_history', methods=['GET', 'POST'])
def user_history():
    if request.method == 'POST':
        user_email = request.form.get('user_email')

        if user_email:
            cursor = db.cursor()

            # 사용 기록 조회
            query = f"""
                SELECT h.h_id, h.h_price, h.s_time, h.e_time, h.s_place, h.e_place,
                       p1.lat AS dep_lat, p1.lot AS dep_lot, p2.lat AS dest_lat, p2.lot AS dest_lot
                FROM history h
                INNER JOIN user u ON h.u_num = u.u_num
                INNER JOIN place p1 ON h.s_place = p1.p_id
                INNER JOIN place p2 ON h.e_place = p2.p_id
                WHERE u.u_email = '{user_email}';
            """
            cursor.execute(query)
            result = cursor.fetchall()

            # 총 사용 시간 조회
            total_usage_time_query = f"""
                SELECT 
                    u.u_id AS user_id,
                    SEC_TO_TIME(SUM(TIMESTAMPDIFF(SECOND, h.s_time, h.e_time))) AS total_usage_time
                FROM
                    user u
                JOIN
                    history h ON u.u_num = h.u_num
                WHERE
                    u.u_email = '{user_email}';
            """
            cursor.execute(total_usage_time_query)
            total_usage_time_result = cursor.fetchone()
            total_usage_time = total_usage_time_result[1] if total_usage_time_result and total_usage_time_result[1] else '0:0:0'

            # 총 금액 계산
            total_price_query = f"""
                SELECT SUM(h.h_price) AS total_price
                FROM history h
                INNER JOIN user u ON h.u_num = u.u_num
                WHERE u.u_email = '{user_email}';
            """
            cursor.execute(total_price_query)
            total_price_result = cursor.fetchone()
            total_price = total_price_result[0] if total_price_result and total_price_result[0] else 0

            return render_template('history.html', user_email=user_email, result=result, total_price=total_price, total_usage_time=total_usage_time)

    return render_template('history.html', user_email=None, result=None, total_price=0, total_usage_time='0:0:0')


# 기록 ID에 해당하는 출발지와 도착지 좌표 가져오기
@app.route('/get_path_coordinates/<int:history_id>')
def get_path_coordinates(history_id):
    cursor = db.cursor()
    query = f"""
        SELECT p_dep.lat AS dep_lat, p_dep.lot AS dep_lot, p_dest.lat AS dest_lat, p_dest.lot AS dest_lot
        FROM history h
        JOIN place p_dep ON h.s_place = p_dep.p_id
        JOIN place p_dest ON h.e_place = p_dest.p_id
        WHERE h.h_id = '{history_id}';
    """
    cursor.execute(query)
    result = cursor.fetchone()

    return {'result': result}

@app.route('/top_monthly_usage', methods=['GET', 'POST'])
def top_monthly_usage():
    if request.method == 'POST':
        selected_year = request.form.get('year')
        selected_month = request.form.get('month')

        if selected_year and selected_month:
            cursor = db.cursor()

            # 사용금액이 많은 순으로 Top 10 조회
            query = f"""
                SELECT h.h_id, h.h_price, h.s_time, h.e_time, h.s_place, h.e_place
                FROM history h
                WHERE YEAR(h.s_time) = {selected_year} AND MONTH(h.s_time) = {selected_month}
                ORDER BY h.h_price DESC
                LIMIT 10;
            """
            cursor.execute(query)
            result = cursor.fetchall()

            return render_template('top_monthly_usage.html', result=result, selected_year=selected_year, selected_month=selected_month)

    return render_template('top_monthly_usage.html', result=None, selected_year=None, selected_month=None)


if __name__ == '__main__':
    app.run(debug=True)
