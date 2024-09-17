import numpy as np

g = 9.81
Ts = 0.01

mCart = 0.409
mRight = 0.1458
mLeft = 0.0805
mBelt = 0.045
mHolderP = 0.024
mHolderC = 0.045
mPole = 0.038
mLoad = 0.007
lPole = 0.498
rPole = 0.00305
lLoad = 0.019
rLoad = 0.006
lPoleLoad = 0.5
lOffset = 0.049

M = 11.70903084
fpc = 0.034078
fc = 0.5
Fspwm = 0.025

if __name__ == "__main__":
    mc = mCart + mRight + mLeft + mBelt + 2*mHolderP + mHolderC
    mp = 2 * (mPole + mLoad)
    m = mc + mp

    lPole0 = lPole/2 - lOffset
    lLoad0 = lPole - lLoad/2 - lOffset + (lPoleLoad - lPole)
    l = (2 * lPole0 * mPole + 2 * lLoad0 * mLoad) / (2 * (mPole + mLoad))

    JPole = 1/12 * mPole * (lPole**2 + 3 * rPole**2)
    JLoad = 1/12 * mLoad * (lLoad**2 + 3 * (rLoad**2 + rPole**2))

    J = 2 * (JPole + mPole * lPole0**2 + JLoad + mLoad * lLoad0**2)

    fp = J * fpc

    Fs = Fspwm * M

    ResCartEnc = 0.235 / 4096
    ResPendEnc = 2*np.pi / 4096

    print(mp, l, J, m, fp, fc, g, M, Fs, ResCartEnc, ResPendEnc)


