import matplotlib.pyplot as plt
import numpy as np

def logistic(x, L=1, k=1, x0=0):
    return L / (1 + np.exp(-k * (x - x0)))

def setting():
    x=np.linspace(-6, 6, 200)
    plt.title('Logistic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

x=np.linspace(-6, 6, 200)
colors=['gray','red','orange','blue','green','skyblue']

plt.figure(figsize=(8, 4)) 
for L in range(1,6):
    y=logistic(x, L=L, k=1, x0=0)
    plt.plot(x,y,label=f'Logistic(L={L},k=1,x0=0)',color=colors[L], linewidth=2)
setting()

for k in range(1,6):
    y=logistic(x, L=1, k=k, x0=0)
    plt.plot(x,y,label=f'Logistic(L=1,k={k},x0=0)',color=colors[k], linewidth=2)
setting()

for x0 in range(-2,3):
    y=logistic(x, L=1, k=1, x0=x0)
    plt.plot(x,y,label=f'Logistic(L=1,k=1,x0={x0})',color=colors[x0+2], linewidth=2)
setting()
