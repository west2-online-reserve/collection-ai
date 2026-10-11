student_id = {"0801": "张三", "0802": "李四", "0803": "王五", "0804": "赵六"}

for key in list(student_id.keys()):
    str_key = str(key)
    last_char = str_key[-1]
    last_digit = int(last_char)
    
    if last_digit % 2 == 0:
        del student_id[key]

print(student_id)

