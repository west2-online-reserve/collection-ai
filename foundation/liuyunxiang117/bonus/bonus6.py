def count_numbers(numbers):
    result = {}
    for i in numbers:
        if i in result:
            result[i] += 1
        else:
            result[i] = 1
    return result

numbers = list(map(int,input().split()))

print(count_numbers(numbers))
