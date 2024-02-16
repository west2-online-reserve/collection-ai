import numpy

a = numpy.array(input().split(" ")).astype("int")
t = min(a);
if t > 0:
    a-=numpy.ones(len(a),dtype=int)*t
print(numpy.exp(a) / numpy.sum(numpy.exp(a)))