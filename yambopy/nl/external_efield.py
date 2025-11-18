import sys
import numpy as np
import scipy 
from scipy import special
import math  
from yambopy.units import ha2ev,fs2aut

#def get_Efield_w(freqs,efield):
#    if efield["name"] == "DELTA":
#        efield_w=efield["amplitude"]*np.exp(1j*freqs[:]*efield["initial_time"])
#    if efield["name"] == "QSSIN":
#        efield_w=E_w=-1.0j*(efield['amplitude'])/2 * [sigma]
#    else:
#        print("Fields different from Delta function not implemented yet")
#        sys.exit(0)
#
#    return efield_w


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
        sigma=efield["damping"]/(2.0*(2.0*np.log(2.0))**0.5)
        qssin_peak=sigma*efield["field_peak"]
        W_0 = np.asarray(efield['frequency'][0]).item()
        T_0 = np.pi / W_0 * float(round(W_0 / np.pi * qssin_peak))
        T = 2*np.pi/W_0
        E_w=1
        #math.sqrt(np.pi/2)*sigma*np.exp(-1j*W_0*T_0)*(special.erf((T-T_0)/math.sqrt(2.0)/sigma)+special.erf(T_0/math.sqrt(2.0)/sigma))
        #print(sigma, qssin_peak, W_0, T_0,  T, E_w)
        if order!=0:
            divide_by_field = (-2.0*1.0j/( efield['amplitude']))**order
        elif order==0:
            divide_by_field = 4.0/( efield['amplitude']*np.conj(E_w))
    else:
        raise ValueError("Electric field not implemented in Divide_by_the_Field!")

    return divide_by_field    
#def Divide_by_the_Field(efield,order):
#    if efield['name']=='SIN' or efield['name']=='SOFTSIN':
#        if order !=0:
#            divide_by_field=1.0j*np.power(2.0/efield['amplitude'],order,dtype=np.cdouble)
#        elif order==0:
#            divide_by_field=4.0/np.power(efield['amplitude'],2.0,dtype=np.cdouble)
#
#    elif efield['name'] == 'QSSIN':
#      #  if order!=0:
#      #  # assume T_i are the *same points* used for P_i, and already in a.u.
#      #      t_rel = T_i - T0
#      #      dt = t_rel[1] - t_rel[0]
#      #      omega0 = freqs[i_f]          # fundamental for this run
#
#      #      # local DFT of the field at ω0
#      #      E_t  = efield['amplitude'] * np.exp(-(t_rel**2)/(2*sigma**2)) * np.sin(omega0 * t_rel)
#      #      Ewin = np.sum(E_t * np.exp(-1j * omega0 * t_rel)) * dt
#
#      #      # normalization
#      #      divide_by_field = 1.0 / (Ewin**order)
#
#       # delta_T= T_i[1]-T_i[0]
#       # E_t = efield['amplitude'] * np.exp(-((T_i - T0)**2)/(2*sigma**2)) * np.sin(freqs[i_f] * T_i)
#       # E_win = delta_T*np.sum(E_t[:] * np.exp(-1j * freqs[i_f] * T_rel)) 
#       # print("-2 i /Eo", -2.0*1.0j/(efield['amplitude']))
#       # # Approximate relations/does not work yet
#        if order!=0:
#       #     divide_by_field = 1.0 / (E_win**order)
#       #     print("1/Ewin", 1/E_win)
#       #     print("E_win", E_win)
#       #     #divide_by_field=1.0j*np.power(2.0/efield['amplitude'],order,dtype=np.cdouble)
#            divide_by_field = (-2.0*1.0j/(efield['amplitude']))**order
#        elif order==0:
#            divide_by_field = 4.0/(efield['amplitude'])**2
#
#
#    else:
#        raise ValueError("Electric field not implemented in Divide_by_the_Field!")
#
#    return divide_by_field





