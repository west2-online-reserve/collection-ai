from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:qwe123@127.0.0.1/flask_todolist'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# 定义数据表模型
class TodoItem(db.Model):
    Id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), index=True)
    content = db.Column(db.String(100))
    status = db.Column(db.String(10), index=True)
    add_time = db.Column(db.DateTime, default=datetime.now)
    deadline = db.Column(db.DateTime)

    def to_dict(self):
        todo = {
            "Id": self.Id,
            "title": self.title,
            "content": self.content,
            "status": self.status,
            "add_time": self.add_time,
            "deadline": self.deadline
        }
        return todo


@app.route('/')
def welcome():
    return "welcome to todolist"


# 添加待办事项
@app.route('/todo', methods=['POST'])
def create_todo_item():
    data = request.get_json()
    todo_item = TodoItem(title=data['title'], content=data['content'], deadline=data['deadline'], status="待办")
    db.session.add(todo_item)
    db.session.commit()

    return jsonify(todo_item.to_dict())


# 将待办事项设置为已完成
@app.route('/todo/<int:item_id>', methods=['PUT'])
def update_todo_item(item_id):
    data = request.get_json()
    todo_item = TodoItem.query.filter_by(Id=item_id).first()
    if not todo_item:
        return jsonify({"message": "待办事项不存在"}), 404
    todo_item.status = data['status']
    db.session.commit()
    return jsonify(todo_item.to_dict())


# 查看所有待办事项
@app.route('/todo/todo', methods=['GET'])
def read_todo_items():
    todo_items = TodoItem.query.filter_by(status="待办").all()
    return jsonify([item.to_dict() for item in todo_items])


# 查看所有已完成事项a
@app.route('/todo/done', methods=['GET'])
def read_done_items():
    done_items = TodoItem.query.filter_by(status="已完成").all()
    return jsonify([item.to_dict() for item in done_items])


# 查看所有事项
@app.route('/todo/all', methods=['GET'])
def read_all_items():
    all_items = TodoItem.query.all()
    return jsonify([item.to_dict() for item in all_items])


# 通过id查询事项
@app.route('/todo/<int:item_id>', methods=['GET'])
def read_item_by_id(item_id):
    todo_item = TodoItem.query.filter_by(Id=item_id).first()
    if not todo_item:
        return jsonify({"message": "待办事项不存在"}), 404
    return jsonify(todo_item.to_dict())


# 删除待办事项
@app.route('/todo/<int:item_id>', methods=['DELETE'])
def delete_todo_item(item_id):
    todo_item = TodoItem.query.filter_by(Id=item_id).first()
    if not todo_item:
        return jsonify({"message": "待办事项不存在"}), 404
    db.session.delete(todo_item)
    db.session.commit()
    return jsonify({"message": "删除成功"})


# 删除所有已完成事项
@app.route('/todo/done', methods=['DELETE'])
def delete_done_items():
    done_items = TodoItem.query.filter_by(status="已完成").all()
    for item in done_items:
        db.session.delete(item)
    db.session.commit()
    return jsonify({"message": "删除成功"})


# 删除所有待办事项
@app.route('/todo/todo', methods=['DELETE'])
def delete_todo_items():
    todo_items = TodoItem.query.filter_by(status="待办").all()
    for item in todo_items:
        db.session.delete(item)
    db.session.commit()
    return jsonify({"message": "删除成功"})


# 通过关键字查询事项
@app.route('/todo/search', methods=['GET'])
def search_items():
    keyword = request.args.get('keyword')
    items = TodoItem.query.filter(TodoItem.title.like(f"%{keyword}%")).all()
    return jsonify([item.to_dict() for item in items])


if __name__ == "__main__":
    app.run()
