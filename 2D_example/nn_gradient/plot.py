import matplotlib.pyplot as plt
import numpy as np



for i in range(10):
    if  i != 3 and i != 7:# and i != 8:
        plot = np.loadtxt(str(i))
        plt.plot(plot[1:])


plt.show()
