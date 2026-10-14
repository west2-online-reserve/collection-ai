# a, b, c = int(input()), int(input()), int(input())

# if a >= b and a >= c:
#     print(a)
#     if b >= c:
#         print(b)
#         print(c)
#     else:
#         print(c)
#         print(b)
# elif b >= a and b >= c:
#     print(b)
#     if a >= c:
#         print(a)
#         print(c)
#     else:
#         print(c)
#         print(a)
# elif c >= a and c >= b:
#     print(c)
#     if a >= b:
#         print(a)
#         print(b)
#     else:
#         print(b)
#         print(a)


# def sort(x, y, z):
#     if x < y:
#         x, y = y, x
#     if x < z:
#         x, z = z, x
#     if y < z:
#         y, z = z, y
#     return x, y, z


# def sort(x, y, z):
#     return sorted((x, y, z), reverse=True)

# print(sort(a, b, c))


# for i in range(1, 10):
#     for j in range(1, 10):
#         print(f"{i}*{j}={i*j}", end="\t")
#     print()


# a = str(input())
# a = a.strip()
# if "ol" in a:
#     a = a.replace("ol", "fzu")
# else:
#     a = a
# print("".join(sorted(a, reverse=True)))  # join是用来把列表转换为字符串的，“”中间的内容是用来连接列表中元素的分隔符


# # a = input()#输出的是字符串
# import ast

# a = ast.literal_eval(input())  # 把 "[1, 'hello', 3]" 这样的文本解析成真正的列表！！！
# for i in a[
#     :
# ]:  # []是切片操作，a[:]表示对列表a进行切片，返回一个新的列表，包含a中的所有元素。这样做的目的是为了在遍历列表时不会因为修改原列表而导致遍历出错。!!!
#     if type(i) == str:
#         a.remove(i)
# print(sorted(a, reverse=True))


# students = {
#     "2026001": "张三",
#     "2026002": "李四",
#     "2026003": "王五",
#     "2026004": "赵六",
# }
# a = list(students.keys())
# for i in a:
#     b = int(i)
#     if b % 2 == 0:
#         del students[i]
# print(students)


# class Commodity:
#     def __init__(self, no, name, price, total_num, remain_num):
#         self.__no = no
#         self.__name = name
#         self.__price = price
#         self.__total_num = total_num
#         self.__remain_num = remain_num

#     def display(self):
#         return f"商品编号：{self.__no}\n商品名称：{self.__name}\n商品价格：{self.__price}\n商品总数：{self.__total_num}\n商品剩余数量：{self.__remain_num}"

#     def income(self):
#         return self.__price * (self.__total_num - self.__remain_num)

#     def setdate(self, no, name, price, total_num, remain_num):
#         self.__no = no
#         self.__name = name
#         self.__price = price
#         self.__total_num = total_num
#         self.__remain_num = remain_num


# import random
# from pathlib import Path

# RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
# JOKERS = ["Big Joker", "Little Joker"]
# order = {rank: i for i, rank in enumerate(RANKS + JOKERS)}
# make_deck = RANKS * 4 + JOKERS
# random.shuffle(make_deck)
# player1 = make_deck[:17]
# player2 = make_deck[17:34]
# player3 = make_deck[34:51]
# others = make_deck[51:]
# Path("player1.txt").write_text("\n".join(player1))
# Path("player2.txt").write_text("\n".join(player2))
# Path("player3.txt").write_text("\n".join(player3))
# Path("others.txt").write_text("\n".join(others))


# def count_numbers(nums):
#     counts = {}
#     for n in nums:
#         counts[n] = counts.get(n, 0) + 1
#     return counts


# data = [1, 3, 3, 7, 3, 1, 9, 9, 9, 9, 0]
# print(count_numbers(data))
