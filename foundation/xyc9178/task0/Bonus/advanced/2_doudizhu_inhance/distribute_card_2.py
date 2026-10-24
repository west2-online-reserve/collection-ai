import random

def sort(x):
    li = ["big_joker","small_joker",2,"A","K","Q","J"] + [str(i) for i in range(10,2,-1)]
#    print(li)
    y = []
    for i in li:
        for j in x:
            if i == j:
                y.append(j)
    return y


total = [str(i) for i in range(2,11)]
total += ["A","J","Q","K"]
total *= 4
total += ["big_joker","small_joker"]
#print(total)
random.shuffle(total)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\advanced\2_doudizhu_inhance\others.txt","w") as f:
    b = sort(total[0:3])
    b = " ".join(b)
    f.write(b)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\advanced\2_doudizhu_inhance\player1.txt","w") as f:
    b = sort(total[3:20])
    b = " ".join(b)
    f.write(b)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\advanced\2_doudizhu_inhance\player2.txt","w") as f:
    b = sort(total[20:37])
    b = " ".join(b)
    f.write(b)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\advanced\2_doudizhu_inhance\player3.txt","w") as f:
    b = sort(total[37:54])
    b = " ".join(b)
    f.write(b)