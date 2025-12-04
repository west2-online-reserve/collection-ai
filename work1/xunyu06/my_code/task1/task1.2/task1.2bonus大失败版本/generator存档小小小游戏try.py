# def my_generator():
#     for i in range(3):
#         yield i
# for i in my_generator():
#     print(i)
# gen=my_generator()
# print(next(gen))  # Output: 0
# print(next(gen))  # Output: 1
# print(next(gen))  # Output: 2


# def feb():
#     a,b=0,1
#     while True:
#         yield a
#         a,b=b,a+b
# fib=feb()
# for _ in range(10):
#     print(next(fib))


# l=(x**2 for x in range(10))
# for _ in range(10):
#      print(next(l))
# l=(x**2 for x in range(10))
# for i in l:
#     print(i)

# def demo():
#     x=yield 1
#     yield x
# gen=demo()
# print(next(gen))  # Output: 1
# print(gen.send(10))
# # print(gen.send(20)) 
# # print(gen.send(30))  # This will raise StopIteration

# def accumulator():
#     total = 0
#     while True:
#         x = yield total   # 每次产出当前 total，等待外部发送一个数赋给 x
#         if x is None:     # 用 None 表示结束
#             break
#         total += x

# c = accumulator()
# print(next(c))     # 激活协程，得到初始 total（0）
# print(c.send(10))  # send 10 -> 返回更新前的 total（0），但函数内部 total 变为 10， 下一次 yield 返回 10
# print(c.send(5))   # send 5 -> 返回 10（上次的 total），内部 total 变为 15
# c.send(None)       # 用 None 表示结束并退出循环
import time
import random
import copy

def test():
    initial_state={'turn':0,'step':0}
    # 启动生成器 可通过 send()传入初始状态
    incoming = yield initial_state
    if incoming:
        initial_state=incoming
    
    while initial_state['turn']<=10:
        # 每回合开始yield当前状态，外部send()覆盖
        incoming= yield initial_state
        if incoming:
            initial_state = incoming
        initial_state['turn']+=1
        count=random.randrange(1,6)
        print(f'>>>turn:{initial_state['turn']}<<<')
        initial_state['step']+=count
        print(f"第{initial_state['turn']}回合，走了{count}步")
        print(f'总步数来到{initial_state['step']}')
        time.sleep(1)

        if initial_state['step']>=30:
            print('你成功了')
            break
    yield initial_state
def interactive_run():
    g= test()
    state = next(g)
    saves = {}
    print("启动状态:",state)
    try:
        while True:
            
            print("当前状态:",state)
            print("操作:\n a.继续\n s(n).存档 l(n).读档")
            cmd = input('>').strip().lower()
            if cmd == 'a':
                state = g.send(None)
            elif cmd[0] == 's' and len(cmd)>=4:
                slot = int(cmd[2:(len(cmd)-1)])
                saves[slot]=copy.deepcopy(state)
                print(f"已经在槽{slot}存档",saves[slot])
                state = g.send(None)
            elif cmd[0] == 'l' and len(cmd)>=4:
                slot = int(cmd[2:(len(cmd)-1)]) #slot作为存档号
                if slot not in saves:
                    print(f'槽{slot}没有存档')
                    continue
                loaded =copy.deepcopy(saves[slot])
                print(f"从槽{slot}读档,载入状态",loaded)
                state = g.send(loaded)
            else:
                print("未知命令")
    except StopAsyncIteration:
        print("结束")
if __name__ =='__main__':
    interactive_run()


