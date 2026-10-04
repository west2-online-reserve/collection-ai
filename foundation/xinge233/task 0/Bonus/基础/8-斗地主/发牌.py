from random import shuffle
from pathlib import Path
kards=[]
kards=[*([3,4,5,6,7,8,9,10,"j","q","k","a",2]*4)]
kards.extend(["big_joker","small_joker"])
kards=list(map(str,kards))
shuffle(kards)
base = Path(__file__).parent
with open(base/"player1.txt","w+") as player1:
    player1.write(" ".join(kards[0:17]))
with open(base/"player2.txt","w+") as player2:
    player2.write(" ".join(kards[17:34]))
with open(base/"player3.txt","w+") as player3:
    player3.write(" ".join(kards[34:51]))
with open(base/"other.txt","w+") as other:
    other.write(" ".join(kards[51:54]))

