for x in range(1, 10):
    list1 = []
    for y in range(x,10):
        list1.append(f"{x}*{y}={x*y}")
    print(*list1)