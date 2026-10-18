a = int(input())
b = int(input())
c = int(input())
d = []
if a > b:
    if c > b and c > a:
        print(f"{c} {a} {b}")
    elif c > b and c < a:
        print(f"{a} {c} {b}")
    else:
        print(f"{a} {b} {c}")
else:
    if c > b and c > a:
        print(f"{c} {b} {a}")
    elif c > a and c < b:
        print(f"{b} {c} {a}")
    else:
        print(f"{b} {a} {c}")