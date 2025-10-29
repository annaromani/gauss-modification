# Copyright (c) 2023-2025, Claudio Attaccalite
# All rights reserved.
#
# This file is part of the yambopy project
# Calculate linear response from real-time calculations (yambo_nl)
#
import numpy as np
from yambopy.units import ha2ev,fs2aut, SVCMm12VMm1,AU2VMm1, Junit, EFunit
from yambopy.nl.external_efield import Divide_by_the_Field
from tqdm import tqdm
import scipy.linalg
import sys
import os
from math import floor, ceil
def Coefficients_Inversion(NW,N_samp,P,W,
        i_f, T_0,sigma,
        T_period,T_range,T_step,efield,INV_MODE, tol=1e-8):
    """
    Compute coefficients inversion using various inversion modes.
    see Sec. III in PRB 88, 235113 (2013) 

    Parameters:
        NW (int): Number of components (assumed NW = NX), order of the response functions.
        NX (int): Number of components.
        P (array): real-time polarizations.
        W (array): Frequency components, multiples of the laser frequency.
        T_period (float): Period of time.
        T_range (tuple): Start and end of time range.
        T_step (float): Time step.
        efield (dict): Contains external field information.
        INV_MODE (str): Inversion mode ('full', 'lstsq', 'svd').

    Returns:
        tuple: X_here (inverted coefficients), Sampling (array of time and values).
    """
    #
    # Here we use always NW=NX
    #
    M_size = 2*(NW-1) + 1  # Positive and negative components plut the zero
    if N_samp==-1: N_samp = M_size
    nP_components = NW
    # 
    i_t_start = int(np.round(T_range[0]/T_step)) 
    i_deltaT  = int(np.round(T_period/T_step)/N_samp)


# Memory alloction 
    M      = np.zeros((N_samp, M_size), dtype=np.cdouble)
    X_here = np.zeros(nP_components, dtype=np.cdouble)


# Calculation of  T_i and P_i
    T_i = np.array([(i_t_start + i_deltaT * i) * T_step - efield["initial_time"] for i in range(N_samp)])
    P_i = np.array([P[i_t_start + i_deltaT * i] for i in range(M_size)])
    Sampling = np.column_stack((T_i / fs2aut, P_i))
    f=open("Sampling_pol_%i.txt"%(i_f+1), "w")
    for i in range(M_size):
        f.write(f"{T_i[i]/fs2aut}, {P_i[i]:.2E}\n")
    f.close()

# Build the M matrix

    M[:, 0] = 1.0
    for i_n in range(1, NW):
         if efield["name"] in {"SIN", "SOFTSIN", "ANTIRES"}:
             exp_neg = np.exp(-1j * W[i_n] * T_i, dtype=np.cdouble)
             exp_pos = np.exp(1j * W[i_n] * T_i, dtype=np.cdouble)
         if efield["name"] in {"QSSIN"}:
             #sigma=efield["damping"]/(2.0*(2.0*np.log(2.0)**0.5))
             #W_0=efield["frequency"][0]
             #to mimin nint(x) in python
             #def nint(x):
             #    return np.floor(x + 0.5) * (x >= 0) + np.ceil(x - 0.5) * (x < 0)
             #T_0=np.pi/W_0*np.float32(np.real(nint(W_0/np.pi*3.0*sigma)))
    
             exp_neg = np.exp(-1j * W[i_n] * T_i, dtype=np.cdouble)*np.exp(i_n*(T_i-T_0)**2/(2*sigma**2))
             exp_pos = np.exp(1j * W[i_n] * T_i, dtype=np.cdouble)*np.exp(i_n*(T_i-T_0)**2/(2*sigma**2))
         M[:, i_n] = exp_neg
         M[:, i_n - 1 + NW] = exp_pos
    
    output_file = f'M_matrix_F{i_f+1}_{INV_MODE}_n_nm.txt'
    header= "#".join([f"E_0 E_1_0 E_2_0 E_-1_0 E_-2_0"])

    with open(output_file, 'w') as f:
        f.write(header + '\n')
        for line in M:
            np.savetxt(f, line.reshape(1, -1), fmt='%.4f', delimiter=' ')

    if INV_MODE=='full':
        try:
            INV = np.linalg.inv(M)
        except:
            print("Singular matrix!!! standard inversion failed ")
            print("set inversion mode to LSTSQ")
            INV_MODE="lstsq"

    if INV_MODE=='lstsq':
# Least-squares
        I = np.eye(N_samp,M_size)
        INV = np.linalg.lstsq(M, I, rcond=tol)[0]

    if INV_MODE=='svd':
# Truncated SVD
        INV = np.linalg.pinv(M,rcond=tol)

# Calculate X_here
#if SOLV_MODE==['fft']:
#        X_here=np.zeros(nP_components,dtype=np.cdouble)
#        for i_n in range(nP_components):
#            X_here[i_n]=X_here[i_n]+np.sum(INV[i_n,:]*P_i[:])
    
    
    #if SOLV_MODE==['LSReg']:
    X_here=np.zeros(NW,dtype=np.cdouble)
    for i_n in range(nP_components):
        if INV_MODE=='lstsq' or INV_MODE=='lstsq_init':
            X_here[i_n]=X_here[i_n]+np.sum(INV[i_n,:]*P_i[:])
        else:
            X_here[i_n]=X_here[i_n]+np.sum(INV[i_n,:]*P_i[:])
    return X_here,Sampling



def Harmonic_Analysis(nldb, X_order, N_samp=-1, T_range=[-1, -1],prn_Peff=False,INV_MODE="full",prn_Xhi=True):
    """
    Perform harmonic analysis on a dataset.

    Parameters:
        nldb: Input dataset with fields, polarizations, and time points.
        X_order (int): Maximum harmonic order.
        T_range (list): Time range for analysis.
        prn_Peff (bool): Print effective polarizations if True.
        INV_MODE (str): Inversion mode ('full', 'lstsq', 'svd').
        prn_Xhi (bool): Print susceptibilities if True.

    Returns:
        tuple: Frequencies, susceptibilities and conductibilities if prn_Xhi is False.
    """
    # Time series
    time = nldb.IO_TIME_points
    # Time step of the simulation
    T_step = time[1] - time[0]
    # External field of the first run
    efield = nldb.Efield[0]
    # Numer of exteanl laser frequencies
    n_runs = len(nldb.Polarization)
    # Array of polarizations for each laser frequency
    polarization = nldb.Polarization
    # Array of currents for each laser frequency
    current     =nldb.Current
    # check if current has been calculated
    l_eval_current=nldb.l_eval_CURRENT
    
    print (l_eval_current)
    # Harmonic frequencies
    freqs = np.array([efield["freq_range"][0] for efield in nldb.Efield], dtype=np.double)

    print("\n* * * Harmonic analysis * * *\n")

    # Check for valid field type
    #if efield["name"] not in {"SIN", "SOFTSIN", "ANTIRES"}:
    #    print("Harmonic analysis works only with SIN or SOFTSIN fields")
    #    sys.exit(0)

    # Check for single field
    if any(ef["name"] != "none" for ef in nldb.Efield_general[1:]):
        print("Harmonic analysis works only with a single field, please use sum_frequency.py functions")
        sys.exit(0)

    
    print(f"Current is {'present' if l_eval_current else 'not present'}: conductibilities will {'not' if l_eval_current else ''} be calculated")
    print(f"Number of runs: {n_runs}")

    #Max and minimun frequencies
    W_step = min(freqs)
    max_W = max(freqs)
    print(f"Minimum frequency: {W_step * ha2ev:.3e} [eV]")
    print(f"Maximum frequency: {max_W * ha2ev:.3e} [eV]")
    print("freqs [eV] = ",freqs* ha2ev)
    T_period = 2.0 * np.pi / W_step
    print(f"Effective max time period: {T_period / fs2aut:.3f} [fs]")
    if (efield["name"]=="SOFTSIN" or efield["name"]=="SIN"):
        sigma=0.0
        if T_range[0] <= 0.0:
            T_range[0] = time[-1] - T_period
        if T_range[1] <= 0.0:
            T_range[1] = time[-1]
    if  efield["name"]=="QSSIN":
        sigma=efield["damping"]/(2.0*(2.0*np.log(2.0))**0.5)
        #T_0=np.pi/efield["frequency"][0]*float(round(efield["frequency"][0]/np.pi*3.*sigma))
        #print(sigma*ha2ev)
        #print(efield["frequency"][0]*ha2ev)
        #print(T_0/fs2aut)
        #if T_range[0] <= 0.0:
        #    T_range[0] = T_0 - T_period/2
        #if T_range[1] <= 0.0:
        #    T_range[1] = T_0 + T_period/2
    #print(f"Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
    #T_range_initial = np.copy(T_range)
        
    M_size = 2 * X_order + 1  # Positive and negative components plut the zero


    # Polarization response
    X_effective = np.zeros((X_order + 1, n_runs, 3), dtype=np.cdouble)
    Susceptibility = np.zeros((X_order + 1, n_runs, 3), dtype=np.cdouble)
    SamplingP = np.zeros((M_size, 2, n_runs, 3), dtype=np.double)
    Harmonic_Frequency = np.zeros((X_order + 1, n_runs), dtype=np.double)

    # Current response
    if l_eval_current:
        Sigma_effective = np.zeros((X_order + 1, n_runs, 3), dtype=np.cdouble)
        Conductibility = np.zeros((X_order + 1, n_runs, 3), dtype=np.cdouble)
        SamplingJ = np.zeros((M_size, 2, n_runs, 3), dtype=np.double)

    # Generate multiples of each frequency
    for i_order in range(X_order+1):
        Harmonic_Frequency[i_order,:]=i_order*freqs[:]
    loop_on_angles = nldb.n_angles != 0
    loop_on_frequencies = nldb.n_frequencies != 0

    if loop_on_angles:
        print("Loop on angles...")

    if loop_on_frequencies:
        print("Loop on frequencies...")
        
    # Find the Fourier coefficients by inversion
    for i_f in tqdm(range(n_runs)):
        T_0=0.0
        #sigma=0.0
        if  efield["name"]=="QSSIN":
            T_period = 2.0 * np.pi / Harmonic_Frequency[1, i_f]
            T_0=np.pi/Harmonic_Frequency[1, i_f]*float(round(Harmonic_Frequency[1, i_f]/np.pi*6.0*sigma))
            if T_range[0] <= 0.0:
                T_range[0] = T_0 - T_period/2
            if T_range[1] <= 0.0:
                T_range[1] = T_0 + T_period/2
        #print(f"Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
        T_range_initial = np.copy(T_range)
        T_range, out_of_bounds = update_T_range(T_period, T_range_initial, time)
        #print("sigma",sigma/fs2aut)
        
        #if N_samp==-1: N_samp = M_size
        if out_of_bounds:
            print(f"WARNING! Time range out of bounds for frequency: {Harmonic_Frequency[1, i_f] * ha2ev:.3e} [eV]")
        for i_d in range(3):
            X_effective[:, i_f, i_d], SamplingP[:, :, i_f, i_d] = Coefficients_Inversion(
                X_order + 1, N_samp, polarization[i_f][i_d, :],
                Harmonic_Frequency[:, i_f],
                i_f,T_0, sigma,
                T_period, T_range, T_step, efield, INV_MODE)
        if l_eval_current:
            for i_d in range(3):
                Sigma_effective[:,i_f,i_d],SamplingJ[:,:,i_f,i_d] = Coefficients_Inversion(
                    X_order+1, N_samp, current[i_f][i_d,:],
                    Harmonic_Frequency[:,i_f],
                    i_f,T_0,sigma,
                    T_period,T_range,T_step,efield,INV_MODE)
    #print("X_eff i_f=26, i_d=1", X_effective[:,25,0])
           
    # Calculate susceptibilities
    for i_order in range(X_order + 1):
        for i_f in range(n_runs):
            if i_order == 1:
                Susceptibility[i_order, i_f, 0] = 4.0 * np.pi * np.dot(efield['versor'], X_effective[i_order, i_f, :])
                if l_eval_current:
                    Conductibility[i_order,i_f,0]= 4.0*np.pi*np.dot(efield['versor'][:],Sigma_effective[i_order,i_f,:])
            else:
                Susceptibility[i_order, i_f, :] = X_effective[i_order, i_f, :]
                if l_eval_current:
                    Conductibility[i_order,i_f,:] = Sigma_effective[i_order, i_f, :]

            Susceptibility[i_order, i_f, :] *= Divide_by_the_Field(nldb.Efield[i_f], i_order)
            
           # values = np.column_stack((Susceptibility[i_order, i_f, 0].real, Susceptibility[i_order, i_f, 1].real, Susceptibility[i_order, i_f, 3].real)
           # output_file = f'o.Susceptibility_F{i_f + 1}_order{i_order}'
           # np.savetxt(output_file, values, header=" Chix Chiy Chiz", delimiter=' ', footer="SUsceptibility")

            if l_eval_current:
                Conductibility[i_order,i_f,:] *=Divide_by_the_Field(nldb.Efield[i_f],i_order)
        

    prefix = f'-{nldb.calc}' if nldb.calc != 'SAVE' else ''

    # Reconstruct effective polarization
    if prn_Peff:
        print("Reconstruct effective polarizations...")
        Peff = np.zeros((n_runs, 3, len(time)), dtype=np.cdouble)
        for i_f in tqdm(range(n_runs)):
            for i_d in range(3):
                for i_order in range(X_order + 1):
                    freq_term = np.exp(-1j * i_order * freqs[i_f] * time)
                    Peff[i_f, i_d, :] += X_effective[i_order, i_f, i_d] * freq_term
                    Peff[i_f, i_d, :] += np.conj(X_effective[i_order, i_f, i_d]) * np.conj(freq_term)
        if l_eval_current:
            print("Reconstruct effective current...")
            Jeff = np.zeros((n_runs, 3, len(time)), dtype=np.cdouble)
            for i_f in tqdm(range(n_runs)):
                for i_d in range(3):
                    for i_order in range(X_order + 1):
                        freq_term = np.exp(-1j * i_order * freqs[i_f] * time)
                        Jeff[i_f, i_d, :] += Sigma_effective[i_order, i_f, i_d] * freq_term
                        Jeff[i_f, i_d, :] += np.conj(Sigma_effective[i_order, i_f, i_d]) * np.conj(freq_term)

        print("Print effective polarizations...")
        for i_f in tqdm(range(n_runs)):
            values = np.column_stack((time / fs2aut, Peff[i_f, 0, :].real, Peff[i_f, 1, :].real, Peff[i_f, 2, :].real))
            output_file = f'o{prefix}.YamboPy-pol_reconstructed_F{i_f + 1}'
            np.savetxt(output_file, values, header="[fs] Px Py Pz", delimiter=' ', footer="Reconstructed polarization")

        if l_eval_current:
            print("Print effective currents...")
            for i_f in tqdm(range(n_runs)):
                values = np.column_stack((time / fs2aut, Jeff[i_f, 0, :].real, Jeff[i_f, 1, :].real, Jeff[i_f, 2, :].real))
                output_file = f'o{prefix}.YamboPy-curr_reconstructed_F{i_f + 1}'
                np.savetxt(output_file, values, header="[fs] Jx Jy Jz", delimiter=' ', footer="Reconstructed current")

    #Units of measure rescaling
    '''
    Comments: 
    i_order is the order of harmonics, i.e. omega^{i_order}. Previous code thought i_order to be the same as the order of Electric field.
    This leads to incorrect shift current conductivity.
    Suppose we have a monochromatic light E(t) ~ cos(ometa t).
    Shift current is zero-th order of harmonics, but second order of electric field. 
    J(0 x omega) ~ sigma^(2) E(omega) E(-omega)
    J(omega) ~ sigma^(1) E(omega) + sigma^(3) E(omega) E(omega) E(-omega)
    J(2 omega) ~ sigma^(2) E(omega) E(omega)
    etc...
    '''
    for i_order in range(X_order+1):
        Susceptibility[i_order,:,:]*=get_Unit_of_Measure(i_order) # Mao: not clear if this is correct
        if l_eval_current:
        #     Conductibility[i_order,:,:]*=get_Unit_of_Measure(i_order)
            if i_order == 0: # shift current is special case of second order
                Conductibility[i_order, :, :] *= Junit/(EFunit**2) 
            else:
                Conductibility[i_order, :, :] *= Junit/(EFunit**i_order) # i_order is the order of harmonics omega^{i_order}, not order of E-field.

    ## Write final results
    #if prn_Xhi:
    #    print("Write final results: xhi^1, xhi^2, xhi^3, etc...")
    #    for i_order in range(X_order + 1):
    #        output_file = f'o{prefix}.YamboPy-X_probe_order_{i_order}'
    #    #     Conductibility[i_order,:,:]*=get_Unit_of_Measure(i_order)
    #        if i_order == 0: # shift current is special case of second order
    #            Conductibility[i_order, :, :] *= Junit/(EFunit**2) 
    #        else:
    #            Conductibility[i_order, :, :] *= Junit/(EFunit**i_order) # i_order is the order of harmonics omega^{i_order}, not order of E-field.

    # Write final results
    if prn_Xhi:
        print("Write final results: xhi^1, xhi^2, xhi^3, etc...")
        for i_order in range(X_order + 1):
            output_file = f'o{prefix}.YamboPy-X_probe_order_{i_order}'
            header = "[eV] " + " ".join([f"X/Im(z){i_order} X/Re(z){i_order}" for _ in range(3)])
            values = np.column_stack((freqs * ha2ev, Susceptibility[i_order, :, 0].imag, Susceptibility[i_order, :, 0].real,
                                      Susceptibility[i_order, :, 1].imag, Susceptibility[i_order, :, 1].real,
                                      Susceptibility[i_order, :, 2].imag, Susceptibility[i_order, :, 2].real))
            np.savetxt(output_file, values, header=header, delimiter=' ', footer="Polarization Harmonic analysis results")
        if l_eval_current:
            print("Write final results: sigma^1, sigma^2, sigma^3, etc...")
            for i_order in range(X_order + 1):
                output_file = f'o{prefix}.YamboPy-Sigma_probe_order_{i_order}'
                header = "[eV] " + " ".join([f"S/Im(z){i_order} S/Re(z){i_order}" for _ in range(3)])
                values = np.column_stack((freqs * ha2ev, Conductibility[i_order, :, 0].imag, Conductibility[i_order, :, 0].real,
                                      Conductibility[i_order, :, 1].imag, Conductibility[i_order, :, 1].real,
                                      Conductibility[i_order, :, 2].imag, Conductibility[i_order, :, 2].real))
                np.savetxt(output_file, values, header=header, delimiter=' ', footer="Current Harmonic analysis results")

    else:
        return (freqs, Susceptibility, Conductibility) if l_eval_current else (freqs, Susceptibility)



def Coefficients_Inversion_nm(NW,N_samp,P,W,
        i_f, T_0,sigma,
        T_period,T_range,T_step,efield,INV_MODE, tol=1e-8):
    """
    Compute coefficients inversion using various inversion modes.
    see Sec. III in PRB 88, 235113 (2013) 

    Parameters:
        NW (int): Number of components (assumed NW = NX), order of the response functions.
        NX (int): Number of components.
        P (array): real-time polarizations.
        W (array): Frequency components, multiples of the laser frequency.
        T_period (float): Period of time.
        T_range (tuple): Start and end of time range.
        T_step (float): Time step.
        efield (dict): Contains external field information.
        INV_MODE (str): Inversion mode ('full', 'lstsq', 'svd').

    Returns:
        tuple: X_here (inverted coefficients), Sampling (array of time and values).
    """
    #
    # Here we use always NW=NX
    N=NW-1
    #if efield["name"] in {"SIN", "SOFTSIN", "ANTIRES"}:
       # M_size=2*N+1
    #if efield["name"] in {"QSSIN"}:
    #if N%2==0:
    #    M_size =(N*(N+4)+2)/2
    #else:
    #    M_size =(N*(N+4)+1)/2
    M_size = 1+2*N + int(N*(N-1)/2) 
    # Positive and negative components plut the zero
    M_size=int(M_size)
    L = [int(N/2*((N/2+1))),int(((N+1)/2)**2)]
    print(L)
    if N_samp==-1: N_samp = M_size
    if N_samp < M_size:
        print("too few sampling points!")
        T_samp=M_size

    nP_components = NW
    i_t_start = int(np.round(T_range[0]/T_step))
    i_deltaT  = int(np.round(T_period/T_step)/N_samp)


# Memory alloction 
    M      = np.zeros((N_samp, M_size), dtype=np.cdouble)
    X_here = np.zeros(nP_components, dtype=np.cdouble)


# Calculation of  T_i and P_i
    T_i = np.array([(i_t_start + i_deltaT * i) * T_step - efield["initial_time"] for i in range(N_samp)])
    P_i = np.array([P[i_t_start + i_deltaT * i] for i in range(N_samp)])
    #print("P_i", P_i[0], P_i[1], P_i[2])
    #print(P_i)
    Sampling = np.column_stack((T_i / fs2aut, P_i))
    #print(np.shape(Sampling))
    Sampling=np.array(Sampling)
    f=open("Sampling_pol_%i.txt"%(i_f+1), "w")
    for i in Sampling:
        f.write(f"{i[0]} {i[1]} \n")
    f.close()

# Build the M matrix
    I=np.zeros((M_size,2))
    M[:, 0] = 1.0
    j=0
    I[j]=(0,0)
    numbers = range(1, N+1)
    evens = (n for n in numbers if n % 2 == 0)
    odd = (n for n in numbers if n % 2 == 1)
    for n in evens:
        j=j+1
        if efield["name"] in {"QSSIN"}:
            x=1.0 #e**x =  for x = 0
            #M[:,j]=np.exp(n*(T_i-T_0)**2/(2*sigma**2))
        if efield["name"] in {"SIN", "SOFTSIN", "ANTIRES"}:
            x=0.0
            #M[:,j]=1.0
        M[:,j]=np.exp(x*n*(T_i-T_0)**2/(2*sigma**2))
        I[j]=(n, n/2)
        #j=j+1
        #I[j]=(-n, n/2)
    for n in numbers:
       # print("n", n)
        for m in range(0, ceil(n/2)):
            if efield["name"] in {"SIN", "SOFTSIN", "ANTIRES"}:
                x=0 #e**x = 1 for x = 0
            if efield["name"] in {"QSSIN"}:
                x=1
            j=j+1
            M[:, j]=np.exp(-1j * W[n-2*m] * T_i)*np.exp(x*n*(T_i-T_0)**2/(2*sigma**2))
            I[j]= (n,m)
            M[:, int(j+L[N%2])]=np.exp(1j * W[n-2*m] * T_i)*np.exp(x*n*(T_i-T_0)**2/(2*sigma**2))
            I[int(j+L[N%2])]= (-n,m)
    #print(I)
    #print(M)     
#            for i_m in range(0, floor(i_n/2)+1):
#                j=j+1
#                exp_neg = np.exp(-1j * W[i_n-2*i_m] * T_i, dtype=np.cdouble)
#                exp_pos = np.exp(1j * W[i_n-2*i_m] * T_i, dtype=np.cdouble)
#                M[:, j] = exp_neg
#                M[:, j-1+L] = exp_p
#
#    if efield["name"] in {"QSSIN"}:
#        j=0
#        for i_n in range(1, N+1):
#            for i_m in range(0, floor(i_n/2)+1):
#                j=j+1
#                exp_neg = np.exp(-1j * W[i_n-2*i_m] * T_i, dtype=np.cdouble)*np.exp(i_n*(T_i-T_0)**2/(2*sigma**2))
#                exp_pos = np.exp(1j * W[i_n-2*i_m] * T_i, dtype=np.cdouble)*np.exp(i_n*(T_i-T_0)**2/(2*sigma**2))
#                M[:, j] = exp_neg
#                M[:, j-1+L] = exp_pos

    #output_file = f'M_matrix_F{i_f+1}_{INV_MODE}.txt'
    #header= "#".join([f"E_0 E_1_0 E_2_0 E_2_1 E_-1_0, E_-2_0 E_-2_1"])
    #with open(output_file, 'w') as f:
    #    f.write(header + '\n') 
    #    for line in M:
    #        np.savetxt(f, line.reshape(1, -1), fmt='%.4f', delimiter=' ')

    if INV_MODE=='full':
        if M_size != N_samp:
            print("M is not square so switch to least square method")
            INV_MODE='lstsq'
        else: 
            #INV = np.linalg.inv(M)
            COND = np.linalg.cond(M)
            spacing=1/np.spacing(1)
            #print("cond=", COND, "spacing=", spacing)
            if COND < spacing:
                INV = np.linalg.inv(M)
            if  COND > spacing:
                print("Singular matrix!!! standard inversion failed ")
                print("set inversion mode to LSTSQ")
                INV_MODE="lstsq"

    if INV_MODE=='lstsq':
# Least-squares
        INV = np.linalg.lstsq(M, P_i, rcond=tol)[0]

    if INV_MODE=='svd':
# Truncated SVD
        INV = np.linalg.pinv(M,rcond=tol)

# Calculate X_here
#if SOLV_MODE==['fft']:
#        X_here=np.zeros(nP_components,dtype=np.cdouble)
#        for i_n in range(nP_components):
#            X_here[i_n]=X_here[i_n]+np.sum(INV[i_n,:]*P_i[:])

    #f=open("INV_%i.txt"%(i_f+1), "w")
    #for i in INV:
    #    f.write(f"{i} \n")
    #f.close()
    #if SOLV_MODE==['LSReg']:
    n_x=L[N%2]+N//2+1
    X_here=np.zeros(n_x,dtype=np.cdouble)
    for i_n in range(n_x):
        if INV_MODE=='lstsq' or INV_MODE=='lstsq_init':
            X_here[i_n]=INV[i_n]
        else:
            X_here[i_n]=X_here[i_n]+np.sum(INV[i_n,:]*P_i[:]) 
    return X_here,Sampling, I





def Harmonic_Analysis_nm(nldb, X_order=4, N_samp=-1, Sampling_time=1.0, Sampling_range=[1, 1], prn_Peff=False,INV_MODE="full",prn_Xhi=True):
    """
    Perform harmonic analysis on a dataset.

    Parameters:
        nldb: Input dataset with fields, polarizations, and time points.
        X_order (int): Maximum harmonic order.
        T_range (list): Time range for analysis.
        prn_Peff (bool): Print effective polarizations if True.
        INV_MODE (str): Inversion mode ('full', 'lstsq', 'svd').
        prn_Xhi (bool): Print susceptibilities if True.

    Returns:
        tuple: Frequencies, susceptibilities and conductibilities if prn_Xhi is False.
    """
    # Time series
    time = nldb.IO_TIME_points
    # Time step of the simulation
    T_step = time[1] - time[0]
    # External field of the first run
    efield = nldb.Efield[0]
    # Numer of exteanl laser frequencies
    n_runs = len(nldb.Polarization)
    # Array of polarizations for each laser frequency
    polarization = nldb.Polarization
    # Array of currents for each laser frequency
    current     =nldb.Current
    # check if current has been calculated
    l_eval_current=nldb.l_eval_CURRENT
    ##print(efield) 
    # Harmonic frequencies
    N_samp=int(N_samp)
    freqs = np.array([efield["freq_range"][0] for efield in nldb.Efield], dtype=np.double)

    print("\n* * * Harmonic analysis * * *\n")

    # Check for valid field type
    #if efield["name"] not in {"SIN", "SOFTSIN", "ANTIRES"}:
    #    print("Harmonic analysis works only with SIN or SOFTSIN fields")
    #    sys.exit(0)

    # Check for single field
    if any(ef["name"] != "none" for ef in nldb.Efield_general[1:]):
        print("Harmonic analysis works only with a single field, please use sum_frequency.py functions")
        sys.exit(0)

    
    print(f"Current is {'present' if l_eval_current else 'not present'}: conductibilities will {'not' if l_eval_current else ''} be calculated")
    print(f"Number of runs: {n_runs}")

    #Max and minimun frequencies
    W_step = min(freqs)
    max_W = max(freqs)
    print(f"Minimum frequency: {W_step * ha2ev:.3e} [eV]")
    print(f"Maximum frequency: {max_W * ha2ev:.3e} [eV]")
    print("freqs [eV] = ",freqs* ha2ev)
    T_period = 2.0 * np.pi / W_step
    print(f"Effective max time period: {T_period / fs2aut:.3f} [fs]")
   # if (efield["name"]=="SOFTSIN" or efield["name"]=="SIN"):
   #     sigma=1
   #     if Sampling_range[0] <= 0.0:
   #         T_range[0] = time[-1] - T_period
   #     if T_range[1] <= 0.0:
   #         T_range[1] = time[-1]
   # if  efield["name"]=="QSSIN":
   #     sigma=efield["damping"]/(2.0*(2.0*np.log(2.0))**0.5)
        #T_0=np.pi/efield["frequency"][0]*float(round(efield["frequency"][0]/np.pi*3.*sigma))
        #print(sigma*ha2ev)
        #print(efield["frequency"][0]*ha2ev)
        #print(T_0/fs2aut)
        #if T_range[0] <= 0.0:
        #    T_range[0] = T_0 - T_period/2
        #if T_range[1] <= 0.0:
        #    T_range[1] = T_0 + T_period/2
    #print(f"Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
    #T_range_initial = np.copy(T_range)
        
    #if X_order%2==0: M_size =(X_order*(X_order+4)+2)/2
    #else:
    #    M_size =(X_order*(X_order+4)+1)/2 
    N=X_order
    M_size = 1+2*N + int(N*(N-1)/2)
    M_size=int(M_size)
    if N_samp<1: N_samp=M_size
    L = [int(N/2*((N/2+1))),int(((N+1)/2)**2)]
    # Polarization response
    n_x=L[N%2]+N//2+1 # dimension of the retuned response vecotr X
    X_effective = np.zeros((n_x, n_runs, 3), dtype=np.cdouble)
    #X_matrix = np.zeros((X_order + 1, X_order + 1, n_runs, 3 ), dtype=np.cdouble)
    Susceptibility = np.zeros((n_x, n_runs, 3), dtype=np.cdouble)
    SamplingP = np.zeros((N_samp, 2, n_runs, 3), dtype=np.double)
    Harmonic_Frequency = np.zeros((X_order+1, n_runs), dtype=np.double)

    # Current response
    if l_eval_current:
        Sigma_effective = np.zeros((n_x, n_runs, 3), dtype=np.cdouble)
        #Sigma_matrix = np.zeros((X_order + 1, X_order + 1, n_runs, 3 ), dtype=np.cdouble)
        Conductibility = np.zeros((n_x, n_runs, 3), dtype=np.cdouble)
        SamplingJ = np.zeros((N_samp, 2, n_runs, 3), dtype=np.double)

    # Generate multiples of each frequency
    for i_order in range(X_order+1):
        Harmonic_Frequency[i_order,:]=i_order*freqs[:]
    loop_on_angles = nldb.n_angles != 0
    loop_on_frequencies = nldb.n_frequencies != 0

    if loop_on_angles:
        print("Loop on angles...")

    if loop_on_frequencies:
        print("Loop on frequencies...")
    # Find the Fourier coefficients by inversion
    for i_f in tqdm(range(n_runs)):
        T_0=0.0
        T_range=[0,0]
        if (efield["name"]=="SOFTSIN" or efield["name"]=="SIN"):
            sigma=1
            if Sampling_range == [1,1]:
                T_range[0] = time[-1] - T_period
                T_range[1] = time[-1]

        if  efield["name"]=="QSSIN":
            sigma=efield["damping"]/(2.0*(2.0*np.log(2.0))**0.5)
            T_period = 2.0 * np.pi / Harmonic_Frequency[1, i_f]
            qssin_peak=sigma*efield["field_peak"]
            freq = Harmonic_Frequency[1, i_f].item()
            T_0 = np.pi / freq * float(round(freq / np.pi * qssin_peak))
            #T_0=np.pi/Harmonic_Frequency[1, i_f]*float(round(Harmonic_Frequency[1, i_f]/np.pi*qssin_peak))
            #T_0=qssin_peak
            print("T_0",T_0/fs2aut)
            print("T_period", T_period/fs2aut)
            if Sampling_time == 1:
                Sampling_time_aut=Sampling_time
                print(f'setting sampling time around T0={T_0/fs2aut:.1f}[fs]')
                if Sampling_range == [1, 1]:
                    print(f'setting sampling range to one period of oscillation around the peak T0')
                else:
                    print(f'setting sampling range to {-Sampling_range[0]*T_period/2/fs2aut:.2f} - {Sampling_range[1]*T_period/2/fs2aut:.2f} respect the peak in T0')
            else:
                print(f'setting sampling time around {Sampling_time:.1f}[fs]')
                Sampling_time_aut=Sampling_time*fs2aut
                T_0=1
                if Sampling_range == [1, 1]:
                    print("setting sampling range to one period os oscillation around the sampling time")
                else:
                    print(f'setting sampling range to {-Sampling_range[0]*T_period/2/fs2aut:.2f}-{Sampling_range[1]*T_period/2/fs2aut:.2f} respect the sampling time')
            T_range[0] = Sampling_time_aut*T_0 - (Sampling_range[0]*T_period/2)
            T_range[1] = Sampling_time_aut*T_0 + (Sampling_range[1]*T_period/2)
        print(f"Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
        T_range_initial = np.copy(T_range)
        T_range, out_of_bounds = update_T_range(T_period, T_range_initial, time)
        
        #if N_samp==-1: N_samp = M_size
        if out_of_bounds:
            print(f"WARNING! Time range out of bounds for frequency: {Harmonic_Frequency[1, i_f] * ha2ev:.3e} [eV]")
            print(f'Sampling range redefined as the last period of the simulation time')
            print(f"updated Time range: {T_range[0] / fs2aut:.3f} - {T_range[1] / fs2aut:.3f} [fs]")
        for i_d in range(3):
            X_effective[:, i_f, i_d], SamplingP[:,:, i_f, i_d], I = Coefficients_Inversion_nm(
                X_order + 1, N_samp, polarization[i_f][i_d, :],
                Harmonic_Frequency[:, i_f],
                i_f,T_0, sigma,
                T_period, T_range, T_step, efield, INV_MODE)

        if l_eval_current:
            for i_d in range(3):
                Sigma_effective[:,i_f,i_d],SamplingJ[:,:, i_f,i_d], I = Coefficients_Inversion_nm(
                    X_order+1, N_samp, current[i_f][i_d,:],
                    Harmonic_Frequency[:,i_f],
                    i_f,T_0,sigma,
                    T_period,T_range,T_step,efield,INV_MODE)
        
            
    for i_f in range(n_runs):
        for j in range(n_x): 
            # Calculate susceptibilities
            if N == 1:
                Susceptibility[j, i_f, 0] = 4.0 * np.pi * np.dot(efield['versor'],X_effective[j,i_f,:])
                if l_eval_current:
                    Conductibility[j,i_f,0]= 4.0*np.pi*np.dot(efield['versor'][:],Sigma_effective[j,i_f,:])
            else:
                Susceptibility[j, i_f, :] = X_effective[j, i_f, :]
                if l_eval_current:
                    Conductibility[j,  i_f,:] = Sigma_effective[j, i_f, :]

            Susceptibility[j, i_f, :] *= Divide_by_the_Field(nldb.Efield[i_f],  I[j][0])
            if l_eval_current:
                Conductibility[j,i_f,:] *=Divide_by_the_Field(nldb.Efield[i_f], I[j][0])
        
       # output_file = f'Xeffective_F{i_f+1}_{INV_MODE}.txt'
       # header= "#".join([f"X_eff_x   X_eff_y   X_eff_z"])
       # with open(output_file, 'w') as f:
       #     f.write(header + '\n')
       #     for line in X_effective[:,i_f, :]:
       #         np.savetxt(f, line.reshape(1, -1), fmt='%.4E', delimiter=' ')

    #print(X_effective[3, :, 1])
    prefix = f'-{nldb.calc}' if nldb.calc != 'SAVE' else ''

    for j in range(n_x):
        Susceptibility[j,:,:]*=get_Unit_of_Measure(I[j][0])     
        
    
    if prn_Xhi:
        print("Write final results: xhi^1, xhi^2, xhi^3, etc...")
        for j in range(n_x):
            output_file = f'o{prefix}.YamboPy-X_probe_order_{int(I[j][0])}_{int(I[j][1])}_Sampling[-{Sampling_range[0]}:{Sampling_range[1]}]Tperiod_Nmax{N}'
            header = "[eV] " + " ".join([f"X/Im(z) X/Re(z)" for _ in range(3)])
            values = np.column_stack((freqs * ha2ev, Susceptibility[j, :, 0].imag, Susceptibility[j, :, 0].real,
                                     Susceptibility[j, :, 1].imag, Susceptibility[j, :, 1].real,
                                     Susceptibility[j, :, 2].imag, Susceptibility[j, :, 2].real))
            np.savetxt(output_file, values, header=header, delimiter=' ', footer="Polarization Harmonic analysis results")
       # if l_eval_current:
       #     print("Write final results: sigma^1, sigma^2, sigma^3, etc...")
       #     for i_order in range(X_order + 1):
       #         output_file = f'o{prefix}.YamboPy-Sigma_probe_order_{i_order}'
       #         header = "[eV] " + " ".join([f"S/Im(z){i_order} S/Re(z){i_order}" for _ in range(3)])
       #         values = np.column_stack((freqs * ha2ev, Conductibility[i_order, :, 0].imag, Conductibility[i_order, :, 0].real,
       #                               Conductibility[i_order, :, 1].imag, Conductibility[i_order, :, 1].real,
       #                               Conductibility[i_order, :, 2].imag, Conductibility[i_order, :, 2].real))
       #         np.savetxt(output_file, values, header=header, delimiter=' ', footer="Current Harmonic analysis results")

    else:
        return (freqs, Susceptibility, Conductibility) if l_eval_current else (freqs, Susceptibility)

def get_Unit_of_Measure(i_order):
    """
    Calculates the unit of measure based on the order provided.

    Parameters:
    i_order (int): The order that determines the calculation of the unit of measure.

    Returns:
    float: The calculated unit of measure.
    """
    # Define the constant ratio for conversion
    ratio = SVCMm12VMm1 / AU2VMm1

    # Calculate the unit of measure based on the order
    if i_order == 0:
        return ratio
    return np.power(ratio, i_order - 1, dtype=np.float64)

def update_T_range(t_period, t_range_initial, time):
    """
    Updates the time range for analysis based on the specified period.
    
    Parameters:
    t_period (float): The period for the analysis range.
    t_range_initial (list or tuple): Initial time range as [start, end].
    time (list or array): Array of time points for reference.

    Returns:
    tuple: Updated time range as [start, end], and a boolean indicating if the range is out of bounds.
    """
    # Initialize the analysis range
    t_range = list(t_range_initial)  # Ensure we work on a mutable copy
    #t_range[1] = t_range[0] + t_period

    # Check if the range goes out of bounds and adjust if necessary
    t_range_out_of_bounds = False
    if t_range[1] > time[-1]:
        t_range[1] = time[-1]
        t_range[0] = t_range[1] - t_period
        t_range_out_of_bounds = True
    return t_range, t_range_out_of_bounds

