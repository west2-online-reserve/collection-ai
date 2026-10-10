for i in range(1,10):
    for j in range (1,i+1):
        print(f"{j}*{i}={j*i}",end=" ")
        if (j*i)/10 >= 1:
            print(" ",end="")
        else:
            print("  ",end="")
    print("")        
