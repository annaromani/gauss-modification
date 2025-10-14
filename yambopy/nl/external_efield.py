import sys
import numpy as np
import scipy 
from scipy import special
import math  

def get_Efield_w(freqs,efield):
    if efield["name"] == "DELTA":
        efield_w=efield["amplitude"]*np.exp(1j*freqs[:]*efield["initial_time"])
    else:
        print("Fields different from Delta function not implemented yet")
        sys.exit(0)

    return efield_w

    
def Divide_by_the_Field(efield,order):
    
    if efield['name']=='SIN' or efield['name']=='SOFTSIN':
        if order !=0:
            divide_by_field=np.power(-2.0*1.0j/efield['amplitude'],order,dtype=np.cdouble)
        elif order==0:
            divide_by_field=4.0/np.power(efield['amplitude'],2.0,dtype=np.cdouble)

    elif efield['name'] == 'QSSIN':
        # Approximate relations/does not work yet
        sigma=efield["damping"]/(2.0*(2.0*np.log(2.0)**0.5))
        if order!=0:
            divide_by_field = (-2.0*1.0j/(efield['amplitude']))**order
        elif order==0:
            divide_by_field = 4.0/(efield['amplitude'])
    else:
        raise ValueError("Electric field not implemented in Divide_by_the_Field!")

    return divide_by_field





