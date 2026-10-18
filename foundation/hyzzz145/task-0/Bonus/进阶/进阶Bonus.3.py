a = [[1 for _ in range(10)] for _ in range(5)]
def m(list):
    len_1 = len(list)
    len_2 = len(list[1])
    b = []
    c = []
    for i in range(len_2):
        for j in range(len_1):
            b.append(list[j][i])
        c.append(b)
        b.clear()
        #b = []
    return c
a_1 = m(a)
print(a_1)  