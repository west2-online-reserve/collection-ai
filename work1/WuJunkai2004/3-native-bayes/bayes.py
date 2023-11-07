import numpy

import cv
import param
import read
import view

for i in read.get_num_image(6):
    six = cv.image(i)
    print(six.get_image_info())

    new = six.clip_image(six.get_image_info()) 
    new = new.center_image()

    density = [
        new.get_region(0, 0).get_density(),
        new.get_region(0, 1).get_density(),
        new.get_region(1, 0).get_density(),
        new.get_region(1, 1).get_density(),
        new.get_region(2, 0).get_density(),
        new.get_region(2, 1).get_density()
    ]

    print(density)

    density = cv.get_absolute_density(density)
    print( density )
    density = cv.get_visualise_density(density)
    print( density )

    input()
