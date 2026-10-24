import random

total = [i for i in range(2,11)]
total += ["A","J","Q","K"]
total *= 4
total += ["big_joker","small_joker"]
random.shuffle(total)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\fundamental\8_doudizhu\others.txt","w") as f:
    a = " ".join(str(x) for x in total[0:3])
    f.write(a)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\fundamental\8_doudizhu\player1.txt","w") as f:
    a = " ".join(str(x) for x in total[3:20])
    f.write(a)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\fundamental\8_doudizhu\player2.txt","w") as f:
    a = " ".join(str(x) for x in total[20:37])
    f.write(a)
with open(r"collection-ai\foundation\xyc9178\task0\Bonus\fundamental\8_doudizhu\player3.txt","w") as f:
    a = " ".join(str(x) for x in total[37:54])
    f.write(a)