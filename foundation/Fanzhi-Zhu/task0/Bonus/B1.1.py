#法一
x,y,z=map(int,input().split())
nums=[x,y,z]
nums.sort(reverse=True)
print(nums[0],nums[1],nums[2])

#法二
x,y,z=map(int,input().split())
if x>y :
    x,y=y,x
if x < z:
    x, z = z, x
if y < z:
    y, z = z, y

print(x, y, z)