import random
def sort(list):
    a = ["大王","小王","2","A","K","Q","J","10","9","8","7","6","5","4","3"]
    b = []
    for i in a:
        for j in range(0,len(list)):
            if list[j] == i:
                b.append(list[j])
    return b

def doudizhu():
    card = [str(i) for i in range(2,11)] + ["J","Q","A","K"]
    card *= 4
    card += ["大王","小王"]
    random.shuffle(card)
    dipai = card[0:3]
    player1 = card[3:20]
    player2 = card[20:37]
    player3 = card[37:54]

    dipai_sort = sort(dipai)
    player1_sort = sort(player1)
    player2_sort = sort(player2)
    player3_sort = sort(player3)

    dipai_str = " ".join(dipai_sort)
    player1_str = " ".join(player1_sort)
    player2_str = " ".join(player2_sort)
    player3_str = " ".join(player3_sort)

    with open('dipai.txt','w',encoding='UTF-8') as f1:
        f1.write(dipai_str)
    with open('player1.txt','w',encoding='UTF-8') as f2:
        f2.write(player1_str)
    with open('player2.txt','w',encoding='UTF-8') as f3:
        f3.write(player2_str)
    with open('player3.txt','w',encoding='UTF-8') as f4:
        f4.write(player3_str)

doudizhu()
print("程序完成")
