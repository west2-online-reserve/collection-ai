n, m = map(int, input().split())

before = [[1 for _ in range(m)] for _ in range(n)]
after = [[before[i][j] for i in range(n)] for j in range(m)]

for row in before:
    print(row)
for row in after:
    print(row)