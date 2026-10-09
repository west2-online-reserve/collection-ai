def normal_sort(nums):
    arr = nums[:]
    arr.sort()
    arr.reverse()
    return arr

def bubble_sort(nums):
    arr = nums[:]
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if arr[j] < arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def insertion_sort(nums):
    arr = nums[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] < key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def main():
    nums = list(map(int, input().split()))

    nums = normal_sort(nums)
    #nums = bubble_sort(nums)
    #nums = insertion_sort(nums)

    print(*nums)

main()







