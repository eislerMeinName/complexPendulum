import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    pwm = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    pwmneg = -pwm
    force = np.array([0.93, 1.4, 2.00, 2.53, 3.18, 3.72, 4.27, 4.79, 5.33])
    forceneg = -force
    pwm2 = np.concatenate((pwm, pwmneg))
    force2 = np.concatenate((force, forceneg))
    z = np.polyfit(pwm, force, 1)
    z2 = np.polyfit(pwm2, force2, 1)
    p = np.poly1d(z)
    p2 = np.poly1d(z2)

    xp = np.linspace(-0.5, 0.5, 100)
    plt.plot(pwm, force, '.', label='points')
    plt.plot(xp, p(xp), '-', label='fit based on points')
    plt.plot(xp, p2(xp), '--', label='symmetric fit')
    plt.plot(xp, z[0]*xp, label='fit based on points without offset')
    print(z)
    plt.legend()
    plt.xlabel('pwm')
    plt.ylabel('force [N]')
    plt.show()
