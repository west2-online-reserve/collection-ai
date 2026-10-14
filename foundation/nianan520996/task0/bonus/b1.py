x,y,z=map(int,input().split())
print(max(x,y,z),x+y+z-max(x,y,z)-min(x,y,z),min(x,y,z))
d=[x,y,z]
d.sort()
print(d[2],d[1],d[0])
d.sort(reverse=True)
print(d[0],d[1],d[2])