import math

import numpy

import param


class count:
        def __init__(self) -> None:
            self.min = None
            self.max = None
        def __call__(self, num):
            self.min = min(self.min, num) if self.min != None else num
            self.max = max(self.max, num) if self.max != None else num

class image:
    def __init__(self, img):
        self.img = img

    def get_image_info(self):
        le2ri = count()
        to2bo = count()
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                if self.img[i][j] > param.threshold.pixel:
                    le2ri(j)
                    to2bo(i)
        unit = max( (le2ri.max - le2ri.min + 1) / param.threshold.ratio.width, 
                    (to2bo.max - to2bo.min + 1) / param.threshold.ratio.height )
        return param._json({
            'unit': unit,
            'x':    to2bo.min,
            'y':    le2ri.min,
            'width' :int(math.ceil(unit * param.threshold.ratio.width)),
            'height':int(math.ceil(unit * param.threshold.ratio.height))
        })

    def clip_image(self, frame):
        result = numpy.zeros((frame.height, frame.width))
        for x in range(frame.height):
            for y in range(frame.width):
                try:
                    result[x][y] = self.img[x + frame.x][y + frame.y]
                except IndexError:
                    pass
        return image(result)

    def center_image(self):
        high = count()
        for i in range(len(self.img)):
            for j in range(len(self.img[i])):
                if self.img[i][j] > param.threshold.pixel:
                    high(i)
        if(high.max - high.min + 1 == len(self.img)):
            return self
        add = int( (len(self.img) - (high.max - high.min + 1)) / 2 )
        result = numpy.zeros((len(self.img), len(self.img[0])))
        for i in range(len(self.img)-add):
            for j in range(len(self.img[i])):
                result[i+add][j] = self.img[i][j]
        return image(result)
    
    def get_region(self, x, y):
        hight = count()
        width = count()
        hight.min = len(self.img) * x // param.threshold.split.height
        hight.max = len(self.img) * (x + 1) // param.threshold.split.height
        width.min = len(self.img[0]) * y // param.threshold.split.width
        width.max = len(self.img[0]) * (y + 1) // param.threshold.split.width
        result = numpy.zeros((hight.max - hight.min, width.max - width.min))
        for i in range(hight.min, hight.max):
            for j in range(width.min, width.max):
                result[i-hight.min][j-width.min] = self.img[i][j]
        return image(result)

    def get_density(self):
        result = 0.0
        for line in self.img:
            for pixel in line:
                result += pixel / 255
        return result
    

def get_absolute_density(array, more = 1):
    max_density = max(array)
    result = []
    for value in array:
#        result.append( 1.00 if((value / max_density)>=more)else (value / max_density) )
        result.append( value / max_density )
    return result

def get_visualise_density(array):
    result = []
    chars = [0.35, 0.25, 0.2, 0.2]
    for idx in range(len(array)):
        for end in range(1,5):
            if array[idx] <= sum(chars[:end]):
                result.append("{}{}".format(idx+1, chr(96+end)))
                break
    return result