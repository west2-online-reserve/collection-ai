# 2 3 4 5 6 7 8 9 10 A J Q K BlackJoker RedJoker
# a bcdefghi j k lmnop q rstuvwxyz
import random
def xipai(lst):
    num=[]
    letter=[]
    for i in lst:
        if type(i)==int:
            num.append(i)
        else:
            letter.append(i)
    num.sort()
    for i in range(4):
        if "J" in letter:
            letter.remove("J")
            letter.append("J")
    for i in range(4):
        if "Q" in letter:
            letter.remove("Q")
            letter.append("Q")
    for i in range(4):
        if "K" in letter:
            letter.remove("K")
            letter.append("K")
    for i in range(4):
        if "A" in letter:
            letter.remove("A")
            letter.append("A")
    lst=num+letter
    for i in range(4):
        if 2 in lst:
            lst.remove(2)
            lst.append(2)
    if "BlackJoker" in lst:
        lst.remove("BlackJoker")
        lst.append("BlackJoker")
    if "RedJoker" in lst:
        lst.remove("RedJoker")
        lst.append("RedJoker")
    return lst
orign_heapq=[2,3,4,5,6,7,8,9,10,"A","J","Q","K"]*4+["BlackJoker","RedJoker"]
lost=[]
for i in range(3):
    index=random.randint(0,len(orign_heapq)-1)
    lost.append(orign_heapq.pop(index))
lost=xipai(lost)
lost=lost[::-1]
with open("others.txt","w") as f:
    for i in lost:
        f.write(str(i)+" ")
random.shuffle(orign_heapq)
lst1=xipai(orign_heapq[0:17])
lst2=xipai(orign_heapq[17:34])
lst3=xipai(orign_heapq[34:51])
lst1,lst2,lst3=lst1[::-1],lst2[::-1],lst3[::-1]
with open("player1.txt","w") as f:
    for i in lst1:
        f.write(str(i)+" ")
with open("player2.txt","w") as f:
    for i in lst2:
        f.write(str(i)+" ")
with open("player3.txt","w") as f:
    for i in lst3:
        f.write(str(i)+" ")