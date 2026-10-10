x, y, z = map(int, input().split())

a = [x, y, z]

a.sort(reverse=True)

print(a[0], a[1], a[2])
#方法二，源自chatGPT