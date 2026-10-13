#[0, name]初始节点
#[1, child]喜欢节点
#本题主要考察超长字符串的处理，能不复制就不复制，因此使用节点完成
n = int(input())
nodes = []
nodes_name = [0] * (n + 1)
#初始化
for i in range(1, n + 1):
    name = input()
    nodes.append((0, name))
    nodes_name[i] = len(nodes) - 1        #初始化时自己对应自己的名字

m = int(input())
for i in range(m):
    u, v = map(int, input().split())
    child = nodes_name[v]
    nodes.append((1, child))
    nodes_name[u] = len(nodes) - 1       #将自己对应的节点转移为当前的child节点

goal_node = nodes_name[1]
sum = 0
while nodes[goal_node][0] == 1:
    sum += 1
    goal_node = nodes[goal_node][1]   #转移到子节点，有点像递归

final_name = nodes[goal_node][1]   #当调用到最后一个节点时，因为对应的nodes[goal_node][0] = 0所以循环没有处理，需要手动处理一次
print("I_love_" * sum + final_name)