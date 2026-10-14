def count(num_list):
    count_dict={}
    for num in num_list:
        if num in count_dict:
            count_dict[num] += 1
        else:
            count_dict[num] = 1
    return count_dict

num_list=input().split()
print(count(num_list))