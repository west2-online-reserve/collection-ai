import time

def dec(f):

    def wrapper(*args,**kwargs):
        start = time.time()
        
        ret = f(*args,**kwargs)
        stop = time.time()
        print (f"函数名:{f.__name__};开始时间:{start};结束时间{stop};耗时:{stop-start}")
        return ret
    
    return wrapper
#测试
# @dec
# def lezi(a):
#     print("乐子"+str(a))
# lezi("yyq")