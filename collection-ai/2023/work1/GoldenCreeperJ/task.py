import re
import time
import pymysql
from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/add', methods=['POST'])
def add():
    try:
        json_data = request.get_json()
        title = json_data.get('title')
        content = json_data.get('content')
        deadtime = json_data.get('deadtime')
        if all((content, title, deadtime)):
            deadtime = list(map(lambda x: f'{x:0>2}', re.split(r'\D+', deadtime.strip())))
            if check_time_format(deadtime) and int(''.join(deadtime)) > int(''.join(list(map(lambda x: f'{x:0>2}', time.localtime()))[:6])):
                deadtime = f"{'-'.join(deadtime[0:3])} {':'.join(deadtime[3:6])}"
                query = "INSERT INTO todolist(title,content,completion_status,add_time,dead_time) VALUES (%s,%s,'undone',now(),%s)"
                args = (title, content, deadtime)
                sql_operate(query=query, args=args)
                return jsonify(code=200, msg='add success',
                               data={'title': title, 'content': content, 'deadtime': deadtime})
            return jsonify(code=200, msg='the time format is wrong')
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/delete/done/all', methods=['DELETE'])
def delete_done_all():
    try:
        query = "delete from todolist where completion_status = 'done'"
        sql_operate(query=query)
        return jsonify(code=200, msg='delete done all success')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/delete/done/one', methods=['DELETE'])
def delete_done_one():
    try:
        json_data = request.get_json()
        id = json_data.get('id')
        if id:
            query = "delete from todolist where id = %s and completion_status = 'done'"
            sql_operate(query=query, args=id)
            return jsonify(code=200, msg='delete done one success')
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/delete/undone/all', methods=['DELETE'])
def delete_undone_all():
    try:
        query = "delete from todolist where completion_status = 'undone'"
        sql_operate(query=query)
        return jsonify(code=200, msg='delete undone all success')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/delete/undone/one', methods=['DELETE'])
def delete_undone_one():
    try:
        json_data = request.get_json()
        id = json_data.get('id')
        if id:
            query = "delete from todolist where id = %s and completion_status = 'undone'"
            sql_operate(query=query, args=id)
            return jsonify(code=200, msg='delete undone one success')
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/modify/to_done/all', methods=['PATCH'])
def delete_to_done_all():
    try:
        query = "update todolist set completion_status ='done' where completion_status = 'undone'"
        sql_operate(query=query)
        return jsonify(code=200, msg='modify to done all success')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/modify/to_done/one', methods=['PUT'])
def modify_to_done_one():
    try:
        json_data = request.get_json()
        id = json_data.get('id')
        if id:
            query = "update todolist set completion_status ='done' where id = %s and completion_status = 'undone'"
            sql_operate(query=query, args=id)
            return jsonify(code=200, msg='modify done one success')
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/modify/to_undone/all', methods=['PATCH'])
def modify_to_undone_all():
    try:
        query = "update todolist set completion_status ='undone' where completion_status = 'done'"
        sql_operate(query=query)
        return jsonify(code=200, msg='modify to undone all success')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/modify/to_undone/one', methods=['PUT'])
def modify_to_undone_one():
    try:
        json_data = request.get_json()
        id = json_data.get('id')
        if id:
            query = "update todolist set completion_status ='undone' where id = %s and completion_status = 'done'"
            sql_operate(query=query, args=id)
            return jsonify(code=200, msg='modify undone one success')
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/view/all', methods=['GET'])
def view_all():
    try:
        query = "select * from todolist"
        result = sql_operate(query=query)
        return jsonify(code=200, msg='view all success', date=result)
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/view/undone', methods=['GET'])
def view_undone():
    try:
        query = "select * from todolist where completion_status = 'undone'"
        result = sql_operate(query=query)
        return jsonify(code=200, msg='view undone success', date=result)
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/view/done', methods=['GET'])
def view_done():
    try:
        query = "select * from todolist where completion_status = 'done'"
        result = sql_operate(query=query)
        return jsonify(code=200, msg='view done success', date=result)
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/view/keyword', methods=['GET'])
def view_keyword():
    try:
        value = request.args.get('value')
        if value:
            value = f'%{value}%'
            query = 'select * from todolist where title like %s or content like %s'
            result = sql_operate(query=query, args=(value, value))
            return jsonify(code=200, msg='view keyword success', date=result)
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


@app.route('/view/id', methods=['GET'])
def view_id():
    try:
        id = request.args.get('id')
        if id:
            query = "select * from todolist where id = %s"
            result = sql_operate(query=query, args=id)
            return jsonify(code=200, msg='view id success', date=result)
        return jsonify(code=200, msg='lose param')
    except Exception as e:
        return jsonify(code=200, msg='there is an unexpected wrong:' + str(e))


def check_time_format(time: list[str]):
    mouth_to_day = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    if len(time) == 6 and '1970' <= time[0] <= '2038' and '01' <= time[1] <= '12' and '01' <= time[2] <= '31' and '00' <= time[3] <= '23' and '00' <= time[4] <= '59' and '00' <= time[5] <= '59':
        if int(''.join(time)) < 20380119031407 and ((not ((int(time[0]) % 400 == 0 or int(time[0]) % 4 == 0 and int(time[0]) % 100) and int(time[1]) == 2) and int(time[2]) <= mouth_to_day[int(time[1])]) or ((int(time[0]) % 400 == 0 or int(time[0]) % 4 == 0 and int(time[0]) % 100 and int(time[1]) == 2) and int(time[2]) > 30)):
            return True
    return False


def sql_operate(query, args=None):
    conn = pymysql.connect(host='localhost', user='root', password='3d1415926', database='mydb')
    cursor = conn.cursor()
    cursor.execute(query=query, args=args)
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    return result


if __name__ == '__main__':
    app.run()
