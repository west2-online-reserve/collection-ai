import param

def print_edges(edges):
    for line in range(28):
        for idx in range(28):
            if (line, idx) in edges:
                print('▩', end='')
            else:
                print('◻', end='')
        print('')


def print_array(array):
    for line in array:
        for idx in line:
            if idx > param.pixel.solid:
                print('▩', end='')
            elif idx > param.pixel.hollow:
                print('◻', end='')
            else:
                print('-', end='')
        print('')


def change_to_list(array):
    new_array = []
    for line in range(28):
        for idx in range(28):
            if array[line][idx] > param.pixel.solid:
                new_array.append((line, idx))
    return new_array


if(__name__ == '__main__'):
    import read
    imgs = read.read_train_images()
    for img in imgs:
        print_array(img.reshape((28, 28)))
        input("=====")