n = int(input())
d = {}
idk = []
for _ in range(n):
    k, v = input().split()
    d[int(k)] = v

for k, v in d.items():
    if k % 10 % 2 == 0:
        idk.append(k)
for k in idk:
    del d[k]

print(d)




