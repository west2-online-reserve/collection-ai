a="\t1\t2\t3\t4\t5\t6\t7\t8\t9\t"
print(a)
for i in range (1,10):
    b=[i]
    for j in range(1,10):
        b.append(i*j)
    print(*b,sep='\t')
     