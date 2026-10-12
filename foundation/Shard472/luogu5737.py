a,b=map(int,input().split())
years = list()
total = 0
for i in range( a , b+1 ):
    if (i % 4 == 0 and i % 100 !=0) or ( i % 400 == 0 ):
        total +=1
        years.append(i)

print(total)
print(*years)

