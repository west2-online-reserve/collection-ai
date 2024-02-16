import read

img = read.read_test_images()

def view(idx):
    picture = img[idx].reshape((28, 28))
    for line in picture:
        for item in line:
            if  (item>255/3*2):
                print('▩',end='')
            elif(item>255/3*1):
                print('◻',end='')
            else:
                print(' ',end='')
        print('')


def print_edges(edges):
    for line in range(28):
        for idx in range(28):
            if (line, idx) in edges:
                print('▩', end='')
            else:
                print('◻', end='')
        print('')


def change_to_list(array, threshold=255/3*2):
    new_array = []
    for line in range(28):
        for idx in range(28):
            if array[line][idx] > threshold:
                new_array.append((line, idx))
    return new_array


if(__name__ == '__main__'):
    while True:
        view(int(input('input index:')))