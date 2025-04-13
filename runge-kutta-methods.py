import numpy as np
import matplotlib.pyplot as plt


def RK2(func, h, x_0, x_max, y_0): # func(x, y) -> [x_vals, y_vals]

    num_points = int((abs(x_0 - x_max)/h) + 1)
    x = np.zeros(num_points)
    y = np.zeros(num_points)
    
    x[0] = x_0
    y[0] = y_0
    F = np.zeros(3, dtype=float)
    n = 0

    while not np.isclose(x[n], x_max):
        x[n+1] = x[n] + h

        F[1] = h * func(x[n], y[n])
        F[2] = h * func(x[n] + h, y[n] + F[1])

        y[n+1] = y[n] + 0.5 * (F[1] + F[2])

        n += 1

    return [x, y]




def RK3(func, h, x_0, x_max, y_0):

    num_points = int((abs(x_0 - x_max)/h) + 1)
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    x[0] = x_0
    y[0] = y_0
    F = np.zeros(4, dtype=float)
    n = 0

    while not np.isclose(x[n], x_max):
        x[n+1] = x[n] + h

        F[1] = h * func(x[n], y[n])
        F[2] = h * func(x[n] + h/2, y[n] + F[1]/2)
        F[3] = h * func(x[n] + 3*h/4, y[n] + 3*F[2]/4)

        y[n+1] = y[n] + (1/9)*(2*F[1] + 3*F[2] + 4*F[3])

        n += 1
    
    return [x, y]



def RK4(func, h, x_0, x_max, y_0):

    num_points = int((np.ceil(x_max - x_0/h)) + 1)
    x = np.zeros(num_points)
    y = np.zeros(num_points)

    x[0] = x_0
    y[0] = y_0
    F = np.zeros(5, dtype=float)
    n = 0

    for n in range(0, num_points-1):
        x[n+1] = x[n] + h

        F[1] = h * func(x[n], y[n])
        F[2] = h * func(x[n] + h/2, y[n] + F[1]/2)
        F[3] = h * func(x[n] + h/2, y[n] + F[2]/2)
        F[4] = h * func(x[n] + h, y[n] + F[3])

        y[n+1] = y[n] + (1/6)*(F[1] + 2*F[2] + 2*F[3] + F[4])

        n += 1
    
    return [x, y]




# def f(x, y):    
#     return (5 * x**2 - y)/(np.exp(x + y))

# vals = RK4(f, 0.1, x_0=0, x_max=10, y_0=1)

# plt.plot(vals[0], vals[1])
# plt.grid()
# plt.show()




        