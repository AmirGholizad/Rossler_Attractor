#Adding necessary libraries
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

#Taking an interval of [t0,tn] for t and discretizating it to N+1 numbers
def Discretizator(N, t0, tn):

	T = np.zeros(N+1)
	#T is an array of numbers for values that t is allowed to have
	H = (tn-t0)/N
	#H is the difference between two numbers inside T
	for i in range(0,len(T)):
		T[i] = t0+(i*H)

	return [H,T]

#Enter the parameters
a = 0.2
b = 0.2
c = 2.5
N = 1000
t0 = 0
tn = 100
t = Discretizator(N, t0, tn)[1]
h = Discretizator(N, t0, tn)[0]

#Define the Runge-Kutta algorithm
def RK4(x0, y0, z0, N, t, h):
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    z = np.zeros(N + 1)
    x[0] = x0
    y[0] = y0
    z[0] = z0

    for n in range(1,N+1):

        K1x = f1(x[n - 1], y[n - 1], z[n - 1], t[n - 1])
        K1y = f2(x[n - 1], y[n - 1], z[n - 1], t[n - 1])
        K1z = f3(x[n - 1], y[n - 1], z[n - 1], t[n - 1])


        K2x = f1(x[n - 1] + K1x * (h / 2),
                 y[n - 1] + K1y * (h / 2),
                 z[n - 1] + K1z * (h / 2), t[n - 1] + h / 2)
        K2y = f2(x[n - 1] + K1x * (h / 2),
                 y[n - 1] + K1y * (h / 2),
                 z[n - 1] + K1z * (h / 2), t[n - 1] + h / 2)
        K2z = f3(x[n - 1] + K1x * (h / 2),
                 y[n - 1] + K1y * (h / 2),
                 z[n - 1] + K1z * (h / 2), t[n - 1] + h / 2)

        K3x = f1(x[n - 1] + K2x * (h / 2),
                 y[n - 1] + K2y * (h / 2),
                 z[n - 1] + K2z * (h / 2), t[n - 1] + h / 2)
        K3y = f2(x[n - 1] + K2x * (h / 2),
                 y[n - 1] + K2y * (h / 2),
                 z[n - 1] + K2z * (h / 2), t[n - 1] + h / 2)
        K3z = f3(x[n - 1] + K2x * (h / 2),
                 y[n - 1] + K2y * (h / 2),
                 z[n - 1] + K2z * (h / 2), t[n - 1] + h / 2)

        K4x = f2(x[n - 1] + K3x * h, y[n - 1] + K3y * h, z[n - 1] + K3z * h, t[n - 1] + h)
        K4y = f1(x[n - 1] + K3x * h, y[n - 1] + K3y * h, z[n - 1] + K3z * h, t[n - 1] + h)
        K4z = f3(x[n - 1] + K3x * h, y[n - 1] + K3y * h, z[n - 1] + K3z * h, t[n - 1] + h)


        x[n] = x[n - 1] + (K1x + 2 * K2x + 2 * K3x + K4x) * h / 6
        y[n] = y[n - 1] + (K1y + 2 * K2y + 2 * K3y + K4y) * h / 6
        z[n] = z[n - 1] + (K1z + 2 * K2z + 2 * K3z + K4z) * h / 6

    return [x,y,z]

#f1 is the righ hand side of the first equation
def f1(x, y, z,t):
    return -y-z

#f2 is the right hand side of the second equation
def f2(x, y, z, t):
    return x+a*y

#f3 is the right hand side of the third equation
def f3(x, y, z, t):
    return b+z*(x-c)

X = np.zeros(N+1)
Y = np.zeros(N+1)
Z = np.zeros(N+1)

#Give the initial conditions
x0 = 2
y0 = 2
z0 = 2

#Calculate X, Y and Z
X = RK4(x0, y0, z0, N, t, h)[0]
Y = RK4(x0, y0, z0, N, t, h)[1]
Z = RK4(x0, y0, z0, N, t, h)[2]


#Plot the results
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.size'] = 10
fig = plt.figure(figsize=(13,9))
ax = fig.gca(projection='3d')
ax.set_ylim(-20, 20)
ax.set_xlim(-20, 20)
ax.set_zlim(0, 30)
ax.view_init(20, 160)
leGend = ' '.join([str(elem) for elem in ['a','=',a,'   ','b','=',b,'   ','c','=',c,'   ']])
ax.plot(X,Y,Z,'k', alpha=0.7, label=leGend)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend(loc=4)
plt.show()
