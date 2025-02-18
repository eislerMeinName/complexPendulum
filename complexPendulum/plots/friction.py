import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', size=16)
M = np.array([11.1666666666666666, -0.2222222222222222222])
Mf = np.poly1d(M)
mcart = 0.45412
g = 9.81
pwms = np.array([0.06, 0.07, 0.08, 0.09,  0.1, 0.11, 0.12, 0.13, 0.14])
endg = np.array([0.17334, 0.30156, 0.466999999999, 0.6024, 0.77224, 0.90306, 1.02352, 1.12106, 1.23544])

z = np.polyfit(endg, Mf(pwms), 1)
print("mu: " + str(z[1] / mcart / g))
print("eps: " + str(z[0]))
p = np.poly1d(z)
print(p)
z2 = np.polyfit(endg, M[0]*pwms, 1)
p2 = np.poly1d(z2)

xp = np.linspace(0, 1.5, 100)

_ = plt.plot(endg, Mf(pwms), '.', label='data points')
plt.plot(xp, p(xp), '-', label='fit')

plt.xlabel('velocity [m/s]')
plt.ylabel('friction force [N]')
plt.ylim(0, 2)
plt.legend()
plt.show()
