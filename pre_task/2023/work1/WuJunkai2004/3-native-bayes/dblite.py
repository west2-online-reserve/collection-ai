# Author: WuJunkai2004
# Update: 2023-11-23
# Version: 1.1.0


import sqlite3

debug   = False
defined = [':memory:', 'debug.db'][debug]


def _unsupport(*args, **kwargs):
    raise RuntimeError('不支持此方法')


class CONLUMN:
    def __init__(self, cursor, table, conlumn):
        self.cursor = cursor
        self.table  = table
        self.name   = conlumn

    def __setitem__(self, __id, __value):
        self.insert(__id, __value)

    def __getitem__(self, __id):
        if(__id < 1):
            raise RuntimeError("下标错误")
        self.cursor.execute("SELECT {} FROM {} WHERE oid={}".format( self.name, self.table, __id))
        return self.cursor.fetchone()[0]

    def update(self, __id, __value):
        self.insert(__id, __value)

    def insert(self, __id, __value):
        self.cursor.execute('UPDATE {} set "{}"="{}" WHERE oid={}'.format(self.table, self.name, __value, __id))
        list.insert(self, __id-1, __value)

    def index(self, __value, __start = 0, __stop = 0x7fffffffffffffff):
        #return list.index(self, __value, __start, __stop) + 1
        self.cursor.execute("SELECT oid FROM {} WHERE {}='{}'".format(self.table, self.name, __value))
        return self.cursor.fetchone()[0]
    
    def __iter__(self):
        self.cursor.execute("SELECT {} FROM {}".format(self.name, self.table))
        for item in self.cursor.fetchall().__iter__():
            yield item[0]


class TABLE:
    def __init__(self, cursor, table) -> None:
        self.cursor = cursor
        self.name   = table

    def __getitem__(self, __name):
        return CONLUMN(self.cursor, self.name, __name)

    def create(self, *__conlumns):
        self.cursor.execute( 'CREATE TABLE {}\n({});'.format(self.name, ',\n'.join(['"{}" TEXT'.format(item) for item in __conlumns]) ) )

    def insert(self, *__value) -> None:
        self.cursor.execute("INSERT INTO {} VALUES({})".format(self.name, ",".join(['"{}"'.format(item) for item in __value]) ) )

    def filder(self, __filder):
        pass



class SQL:
    def __init__(self, file = defined) -> None: 
        self.connect = sqlite3.connect(file)
        self.cursor  = self.connect.cursor()

    def __getitem__(self, __name) -> TABLE:
        return TABLE(self.cursor, __name)
        
    def __del__(self) -> None:
        self.connect.commit()
        self.cursor .close()
        self.connect.close()

    def commit(self):
        self.connect.commit()


def _shell():
    print("欢迎使用 dblite shell")
    while(True):
        cmmd = ''
        line = input('>>>').rstrip()
        cmmd += line
        while(line and (line[0] == ' ' or line[-1] == ':')):
            line = input('...').rstrip()
            cmmd += '\n' + line
        try:
            result = eval(cmmd)
        except Exception as e1:
            try:
                exec(cmmd)
            except Exception as e2:
                print("ERROR ! : {}".format(e2))
        else:
            if(result != None):
                print(result)


if(__name__ == "__main__"):
    _shell()
