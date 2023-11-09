import dblite

db = dblite.SQL("clean.db")

data = {}

for num in range(10):
    name = "num_{}".format(num)
    for i in range(1,7):
        part = "part_{}".format(i)
        for item in db[name][part]:
            if item not in data.keys():
                data[item] = 0
            data[item] += 1

print(data)
input("Press enter to exit")