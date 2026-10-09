import re

def check(password):
    pattern = re.compile(r'[A-Za-z0-9]{6,18}')
    return pattern.fullmatch(password) is not None

def main():
    arr = input()
    print(check(arr))

main()