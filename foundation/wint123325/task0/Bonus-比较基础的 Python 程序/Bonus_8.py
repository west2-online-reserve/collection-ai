card=[]
for i in range(2,11):
    card.append(str(i))
card.append("A")
card.append("J")
card.append("Q")
card.append("K")
card=card+card+card+card
card.append("SMALL JOKER")
card.append("BIG JOKER")
import random
random.shuffle(card)
box={}
for i in range(0,54):
    box[i]=card[i]   
num1=random.sample(list(box.keys()),17)
card1=[]
for i in num1:
    card1.append(box[i])
    del box[i]
num2=random.sample(list(box.keys()),17)
card2=[]
for i in num2:
    card2.append(box[i])
    del box[i]
num3=random.sample(list(box.keys()),17)
card3=[]
for i in num3:
    card3.append(box[i])
    del box[i]
card4=[]    
for i in range(0,55):
    if i in list(box.keys()):
        card4.append(box[i])
A=open("player1.txt","w")
A.write(','.join(card1))
B=open("player2.txt","w")
B.write(','.join(card2))
C=open("player3.txt","w")
C.write(','.join(card3))
D=open("others.txt","w")
D.write(','.join(card4))
print("over")