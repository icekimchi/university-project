from flask import Flask, render_template, request
import pymysql

app = Flask(__name__)

db = pymysql.connect(host="localhost", user="root", passwd="0000", db="camping", charset="utf8")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    location = request.args.get('location') or request.form.get('location') #지역 아이콘 선택 여부 확인
    campground_selected = 'campground' in request.form  # 물놀이장 체크박스 선택 여부 확인
    firewood_selected = 'firewood' in request.form # 장작판매 체크박스 선택 여부 확인
    cursor = db.cursor()

    tables = ['단양', '보은', '영동', '옥천', '음성', '제천', '증평', '진천', '청주', '충주']
    result = []

    if location in tables:
        query = f"SELECT * FROM {location}"
        if campground_selected and firewood_selected:
            query += " WHERE find LIKE '%물놀이장%' AND find LIKE '%장작판매%'"
        elif campground_selected:
            query += " WHERE find LIKE '%물놀이장%'"
        elif firewood_selected:
            query += " WHERE find LIKE '%장작판매%'"
    else:
        for table in tables:
            query = f"SELECT * FROM {table} WHERE name LIKE '%{location}%'"
            if campground_selected and firewood_selected:
                query += " AND find LIKE '%물놀이장%' AND find LIKE '%장작판매%'"
            elif campground_selected:
                query += " AND find LIKE '%물놀이장%'"
            elif firewood_selected:
                query += " AND find LIKE '%장작판매%'"

            cursor.execute(query)
            result += cursor.fetchall()

        return render_template('search.html', result=result)

    cursor.execute(query)
    result = cursor.fetchall()

    return render_template('search.html', result=result)

@app.route('/region/<location>', methods=['GET'])
def region(location):
    cursor = db.cursor()
    query = f"SELECT * FROM {location}"
    cursor.execute(query)
    result = cursor.fetchall()
    return render_template('region.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

