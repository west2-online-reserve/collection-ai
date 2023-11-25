import cv
import dblite
import read
import param

data = {}

db = dblite.SQL('clean.db')

for num in range(10):
    try:
        parts = ["part_{}".format(i) for i in range(1, 1 + param.threshold.split.width * param.threshold.split.height )]
    except:
        pass

    db['num_{}'.format(num)].create( *parts )

    for item in read.get_num_image(num):
        img = cv.image(item)
        img = img.clip_image(img.get_image_info())
        img = img.center_image()

        density = []
        for h in range(param.threshold.split.height):
            for w in range(param.threshold.split.width):
                density.append( img.get_region(h, w).get_density() )


        density = cv.get_absolute_density(density)
        density = cv.get_visualise_density(density)
        db['num_{}'.format(num)].insert( *density )
        for item in density:
            if(item not in data.keys()):
                data[item] = 1
            else:
                data[item] += 1

print(data)

input()