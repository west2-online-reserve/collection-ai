A={2026001:"Tom",2026002:"Jack",2026003:"Peter",2026004:"Ann"}
for i in list(A.keys()):
    if i % 2 == 0:
        del A[i]
print(A)        
        
