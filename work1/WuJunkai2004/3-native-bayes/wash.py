import cv
import dblite
import read

data = {}

db = dblite.SQL('clean.db')

for num in range(10):
    try:
        parts = ["part_{}".format(i) for i in range(1,7)]
        db['num_{}'.format(num)].create( *parts )
    except:
        pass

    for item in read.get_num_image(num):
        img = cv.image(item)
        img = img.clip_image(img.get_image_info())
        img = img.center_image()

        density = [
            img.get_region(0, 0).get_density(),
            img.get_region(0, 1).get_density(),
            img.get_region(1, 0).get_density(),
            img.get_region(1, 1).get_density(),
            img.get_region(2, 0).get_density(),
            img.get_region(2, 1).get_density()
        ]

        density = cv.get_absolute_density(density)
        density = cv.get_visualise_density(density)
        db['num_{}'.format(num)].insert( *density )
        for item in density:
            if(item not in data.keys()):
                data[item] = 1
            else:
                data[item] += 1

print(data)