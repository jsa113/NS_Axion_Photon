# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:40:36 2023

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz
"""

from parameters import *
from functions import *
import autograd.numpy as np
from autograd import grad, jacobian

# Import the appropriate magentic field functions
if mag_field_choice == 'GR':
    if Theta_m == 0:
        from GR_magnetic_field_aligned import *
        # print('Dispersion Method: GR Aligned Magnetic Field Imported')
        
    else:
        from GR_magnetic_field import *
        # print('Dispersion Method: GR Magnetic Field Imported')
        
elif mag_field_choice == 'GJ':
    from GJ_magnetic_field import *
    # print('Dispersion Method: GJ Magnetic Field Imported')

# =============================================================================
# Extra functions
# =============================================================================

def Global_velocity_vector(r, theta, phi):
    """The Global unit velocity vector of the plasma, in this case a static plasma
    
    Arugments:
        r = radius
        theta = polar angle
        phi = azimuthal angle
    Returns:
        The velocity of a static observer
    """
    
    g_ab = metric(r, theta, phi) # metric
    
    return np.array([1 / np.sqrt(-g_ab[0,0]), 0, 0, 0])

# =============================================================================
# Partial derivaties to be used in the dispersion equations
# =============================================================================

dinv_g_dt = np.zeros((4,4)) # partial g_ab / partial t (0 for schwarzschild)
dinv_g_dr = jacobian(inv_metric, argnum=0) #partial g_ab / partial r
dinv_g_dtheta = jacobian(inv_metric, argnum=1) #partial g_ab / partial theta
dinv_g_dphi = np.zeros((4,4)) # partial g_ab / partial phi (0 for schwarzschild)

## Use these if the metric depends on time or phi
# dinv_g_dt = jacobian(inv_metric, argnum=4) #partial g_ab / partial t
# dinv_g_dphi = jacobian(inv_metric, argnum=2) #partial g_ab / partial phi

dU_dt = np.zeros(4)
dU_dr = jacobian(Global_velocity_vector, argnum=0)
dU_dtheta = jacobian(Global_velocity_vector, argnum=1)
dU_dphi = jacobian(Global_velocity_vector, argnum=2)

# =============================================================================
# Define the function to be integrated (the system of ODEs)
# =============================================================================

if dispersion_method == 'vacuum':
    def F(t, y):
        """
        Linear ODE system to be fed into Scipy's solve_ivp.
        
        This system describes how a photon in a dispersive medium will propagate
        given a dispersion relation via the Hamilationian.
        
        This is the relationship for a vacuum.
        
        Arugments:
            t = the solver integration time step
            y = the current steps information. This contains the photon 4-position
            and 4-momentum
        Returns:
            A vector containing:
                dxdl = The step in the 4-position
                dkdl = The step in the 4-momentum
        """
        
        x = y[:4] # 4 vector (ct, r, theta, phi) # x upstairs index 
        k = y[4:] # 4 vector (omega/c, k_r, k_theta, k_phi) # k downstairs index (contravariant vector)
        
        dxdl = 2 * (np.matmul(inv_metric(x[1], x[2], x[3]), k))

        # The partial derivative of the metric wrt coordinate a (g_abc)
        d_a_inv_g = np.array([dinv_g_dt, dinv_g_dr(x[1], x[2], x[3]), dinv_g_dtheta(x[1], x[2], x[3]), dinv_g_dphi]) 
        
        #The second hamilationian equation (dk_a/dl)
        dkdl = - np.einsum('ijk,j,k', d_a_inv_g, k,k) 

        return np.append(dxdl, dkdl) # Return a vector containing the necessary derivatives

elif dispersion_method == 'unmagnetised':
    def F(t, y):
        """
        Linear ODE system to be fed into Scipy's solve_ivp.
        
        This system describes how a photon in a dispersive medium will propagate
        given a dispersion relation via the Hamilationian.
        
        This is the relationship for an unmagnetised plasma.
        
        Arugments:
            t = the solver integration time step
            y = the current steps information. This contains the photon 4-position
            and 4-momentum
        Returns:
            A vector containing:
                dxdl = The step in the 4-position
                dkdl = The step in the 4-momentum
        """
        
        x = y[:4] # 4 vector (ct, r, theta, phi) # x upstairs index 
        k = y[4:] # 4 vector (omega/c, k_r, k_theta, k_phi) # k downstairs index (contravariant vector)
        
        dxdl = 2 * (np.matmul(inv_metric(x[1], x[2], x[3]), k))
    
        # The partial derivative of the metric wrt coordinate a (g_abc)
        d_a_inv_g = np.array([dinv_g_dt, dinv_g_dr(x[1], x[2], x[3]), dinv_g_dtheta(x[1], x[2], x[3]), dinv_g_dphi]) 
        
        # The partial derivative of the plasma frequency squared wrt coordinate a (wp_a)
        d_a_w_p_sq = np.array([dw_p_sq_dt(x[0], x[1], x[2], x[3]), dw_p_sq_dr(x[0], x[1], x[2], x[3]), dw_p_sq_dtheta(x[0], x[1], x[2], x[3]), dw_p_sq_dphi(x[0], x[1], x[2], x[3])])
        
        # The second hamilationian equation (dk_a/dl)
        dkdl = - np.einsum('ijk,j,k', d_a_inv_g, k,k) - d_a_w_p_sq
    
        return np.append(dxdl, dkdl) # Return a vector containing the necessary derivatives

elif dispersion_method == 'magnetised':
    def F(t, y):
        """
        Linear ODE system to be fed into Scipy's solve_ivp.
        
        This system describes how a photon in a dispersive medium will propagate
        given a dispersion relation via the Hamilationian.
        
        For a magnetised plasma.
        
        Arugments:
            t = the solver integration time step
            y = the current steps information. This contains the photon 4-position
            and 4-momentum
        Returns:
            A vector containing:
                dxdl = The step in the 4-position
                dkdl = The step in the 4-momentum
        """
        
        x = y[:4] # 4 vector (t, r, theta, phi) # x upstairs index 
        k = y[4:] # 4 vector (omega, k_r, k_theta, k_phi) # k downstairs index (contravariant vector)
        
        g_ab = metric(x[1], x[2], x[3]) 
        
        b = magnetic_field_unit_vec(x[0], x[1], x[2], x[3])
        
        U = np.array([1 / np.sqrt(-g_ab[0,0]), 0, 0, 0])
        # U = Global_velocity_vector(x[1], x[2], x[3]) # Global velocity in the pulsars rotating frame
        
        W = -1 * np.matmul(k, U) # k^a*U_a
        K_para = np.matmul(k, b) # k^a*b_a
        
        w_p_sq = Omega_p_sq(x[0], x[1], x[2], x[3]) 
        
        dxdl = 2 * (np.matmul(inv_metric(x[1], x[2], x[3]), k) - w_p_sq * (K_para**2 / W**3 * U + K_para / W**2 * b))
    
        # The partial derivative of the metric wrt coordinate a (g_abc)
        d_a_inv_g = np.array([dinv_g_dt, dinv_g_dr(x[1], x[2], x[3]), dinv_g_dtheta(x[1], x[2], x[3]), dinv_g_dphi]) 
        
        # The partial derivative of the plasma frequency squared wrt coordinate a (wp_a)
        d_a_w_p_sq = np.array([dw_p_sq_dt(x[0], x[1], x[2], x[3]), dw_p_sq_dr(x[0], x[1], x[2], x[3]), dw_p_sq_dtheta(x[0], x[1], x[2], x[3]), dw_p_sq_dphi(x[0], x[1], x[2], x[3])])
        
        # The partial derivative of the plasma frequency squared wrt coordinate a (b_a^c)
        d_a_b = np.array([db_dt(x[0], x[1], x[2], x[3]), db_dr(x[0], x[1], x[2], x[3]), db_dtheta(x[0], x[1], x[2], x[3]), db_dphi(x[0], x[1], x[2], x[3])])
    
        #The second hamilationian equation (dk_a/dl)
        dkdl = - np.einsum('ijk,j,k', d_a_inv_g, k,k) + d_a_w_p_sq * (K_para**2 / W**2 - 1) + 2 * w_p_sq * K_para / W**2 * np.matmul(d_a_b, k)
        
        
        # d_a_U = np.array([dU_dt, dU_dr(x[1], x[2], x[3]), dU_dtheta(x[1], x[2], x[3]), dU_dphi(x[1], x[2], x[3])]) 
        #+ 2 * w_p_sq * K_para **2 / W**3 * np.matmul(d_a_U, k) # Extra term, that is very small
        
        return np.append(dxdl, dkdl) # Return a vector containing the necessary derivatives

















