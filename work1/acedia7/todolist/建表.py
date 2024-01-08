from datetime import datetime
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker


SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:qwe123@127.0.0.1/flask_todolist"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = sqlalchemy.orm.declarative_base()

class TodoItem(Base):
    __tablename__ = "todo_item"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(50), index=True)
    content = Column(Text)
    status = Column(String(10), index=True)
    add_time = Column(DateTime, default=datetime.now)
    deadline = Column(DateTime)

Base.metadata.create_all(bind=engine)