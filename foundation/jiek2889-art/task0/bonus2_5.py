import re

password = input()
if re.match(r"^[a-zA-Z0-9]{6,18}$", password):
    print("Good")
else:
    print("Invalid password")
