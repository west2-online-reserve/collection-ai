apple = input()
applelist = apple.split()
newapple=[]
for i in applelist:
    num = int(i)-30
    newapple.append(num)
hand = int(input())
count = 0
for height in newapple:
    if height <= hand:
        count += 1
print(count)

