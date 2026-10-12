def count_number(nums):
    counter = {}
    for num in nums:
        counter[num] = counter.get(num, 0) + 1 #deepseek
    return counter
def main():
    nums = list(map(int, input().split()))
    counter = count_number(nums)
    for num in sorted(counter):
        print(f"{num}: {counter[num]}") 
if __name__ == "__main__":
    main()
