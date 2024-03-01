import numpy

import read
import cv
import param
import dblite


prob = [ dict( zip( param.threshold.word, [0]*24 ) ) for i in range(10) ]
param.other_probability()
#quit()
db = dblite.SQL("clean.db")
for is_num in range(10):
    for has_part in param.threshold.word:
        totol = 0
        goals = 0
        for part in db[ 'num_{}'.format(is_num) ][ 'part_{}'.format(has_part[0]) ]:
            totol += 1
            if part == has_part:
                goals += 1
        prob[is_num][has_part] = goals / totol

#print(prob)


def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)


class Bayes:
    def __init__(self, is_num , has_part):
        self.is_num = is_num
        self.has_part = has_part
        self.probability = prob[is_num][has_part]

    def __call__(self):
        return self.probability * param.probability['is_number_{}'.format(self.is_num)] / param.probability['has_part_{}'.format(self.has_part)]


def check(den):
        result = []
        for i in range(10):
            result.append( 1000 * param.probability["is_number_{}".format(i)] * Bayes(i,den[0])() * Bayes(i,den[1])() * Bayes(i,den[2])() * Bayes(i,den[3])() * Bayes(i,den[4])() * Bayes(i,den[5])() )
        '''print(result)
        print(softmax(result))
        print(numpy.argmax(result))
        print(result.index(max(result)))'''
        return numpy.argmax(result)

if(__name__ == "__main__"):

    schedule = 0

    correct = 0
    wrong = 0

    array = read.read_test_images()
    label = read.read_test_labels()

    for i in range(10000):
        ## check the schedule
        schedule += 1
        if schedule % 100 == 0:
            print(schedule/100)

        ## get the image and density
        den = cv.get_density(array[i].reshape(28,28))

        ## check the result
        result = check(den)
        if result == label[i]:
            correct += 1
        else:
            wrong += 1



    print("correct: {}".format(correct))
    print("wrong: {}".format(wrong))
    print("accuracy: {}".format(correct / (correct + wrong)))

    input()