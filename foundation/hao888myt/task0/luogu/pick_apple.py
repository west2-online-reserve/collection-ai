tree_heights: list[str] = input().split()
taotao_height: str = input()
CHAIR_HEIGHT: int = 30
count: int = 0

for tree_height in tree_heights:
    if int(tree_height) <= int(taotao_height) + CHAIR_HEIGHT:
        count += 1

print(count)