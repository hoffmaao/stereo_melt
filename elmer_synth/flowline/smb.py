# Surface mass balance functions (for upper and lower surfaces)
# Forked from vendor/linear-shelf-melt/nonlinear-model/smb.py. E1b deltas:
# OCEAN-FIXED basal melt anomaly at x = xc (ice advects through it); no
# surface accumulation (through-flow supplies the mass balance).
# Expressions work on both UFL objects and numpy arrays (np.exp(1)** trick).
import numpy as np
from params import m0, stdev, xc


def smb_s(x,t,m0=m0,stdev=stdev,xc=xc):
    # basal melt anomaly (m > 0 = melt), fixed in space at x = xc
    m = m0*(np.exp(1)**(-((x-xc)**2)/(stdev**2)))
    return m

def smb_h(x,t):
    # Surface mass balance at the upper surface: none in the E1b pilot
    return 0.0*x
