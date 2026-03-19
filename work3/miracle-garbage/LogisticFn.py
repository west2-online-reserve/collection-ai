import matplotlib.pyplot as plt
import numpy as np

e = 2.718281828
L = 1
k = 1
x0 = 0

x = np.arange(-10,10)
y = L/(1 + e**(-k * (x - x0)))

plt.figure()
plt.plot(x,y)

plt.show()
input()

plt.figure() 
for i in [-5,0,5]:
    L = i
    y = L/(1 + e**(-k * (x - x0)))
    plt.plot(x,y,label=f'L={i}')
    plt.legend()
L = 1
    
plt.show()
input()

plt.figure()
for i in [-5,0,5]:
    k = i
    y = L/(1 + e**(-k * (x - x0)))
    plt.plot(x,y,label=f'k={i}')
    plt.legend()
k = 1

plt.show()
input()

plt.figure()
for i in [-5,0,5]:
    x0 = i
    y = L/(1 + e**(-k * (x - x0)))
    plt.plot(x,y,label=f'x0={i}')
    plt.legend()
x0 = 1

plt.show()
input()
