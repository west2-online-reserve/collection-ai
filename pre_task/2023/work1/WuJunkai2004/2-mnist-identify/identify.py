import numpy
import read
import view
import queue

LENGTH_THRESHOLD = 8
EDGE_THRESHOLD = 3
PIXEL_THRESHOLD = 96


def check_horisontal(array, x, y):
    node = numpy.zeros((LENGTH_THRESHOLD, 2), dtype=int)
    node[0] = [x, y]

    def check(x, y, floor):
        if(floor == LENGTH_THRESHOLD):
            return array[x][y]
        node[floor] = [x, y] ## 放进新位点
        if(not array[x][y]):
            return False
        if(node[floor-1][0] == x and node[floor-1][1] == y-1):
            return check(x+1, y+1, floor+1) or check(x-1, y+1, floor+1) or check(x, y+1, floor+1)
        else:
            return check(x, y+1, floor+1)

    return check(x, y+1, 1)
    

def check_vertical(array, x, y):
    node = numpy.zeros((LENGTH_THRESHOLD, 2), dtype=int)
    node[0] = [x, y]

    def check(x, y, floor):
        if(floor == LENGTH_THRESHOLD):
            return array[x][y]
        node[floor] = [x, y] ## 放进新位点
        if(not array[x][y]):
            return False
        if(node[floor-1][0] == x-1 and node[floor-1][1] == y):
            return check(x+1, y+1, floor+1) or check(x+1, y-1, floor+1) or check(x+1, y, floor+1)
        else:
            return check(x+1, y, floor+1)
        
    return check(x+1, y, 1)


class point:
    def __init__(self, point):
        self.pos = point
    
    def horison(self):
        deal = queue.Queue()
        deal.put(self.pos)
        while(not deal.empty()):
            pass

    def vertical(self):
        pass


def change_to_bool(array ):
    new_array = numpy.zeros((len(array), len(array[0])), dtype=bool)
    for i in range(len(array)):
        for j in range(len(array[i])):
            new_array[i][j] = array[i][j] > PIXEL_THRESHOLD
    return new_array


def get_edges(array):
    edges = []
    for i in range(len(array)):
        for j in range(len(array[i])):
            if(array[i][j] and int(array[i-1][j])+int(array[i+1][j])+int(array[i][j-1])+int(array[i][j+1]) <= EDGE_THRESHOLD):
                edges.append((i, j))
    return edges


img = read.read_test_images()
pic = img[0].reshape((28, 28))


points = view.change_to_list(pic, PIXEL_THRESHOLD)
pic = change_to_bool(pic)
edges = get_edges(pic)


view.print_edges(points)
print("=====")
view.print_edges(edges)
print("=====")

horisons = []

for item in edges:
    if(check_horisontal(pic, item[0], item[1])):
        horisons.append(item)


view.print_edges(horisons)
print("=====")

verciters = []

for item in edges:
    if(check_vertical(pic, item[0], item[1])):
        verciters.append(item)
view.print_edges(verciters)
