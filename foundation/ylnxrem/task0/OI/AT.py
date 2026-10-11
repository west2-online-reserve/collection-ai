N = int(input())
i = 2
while i < int(N**0.5) + 1:
    if N % i == 0:
        print("no")
        break
    i += 1
else:
    print("yes")