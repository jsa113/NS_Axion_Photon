# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:29:32 2023

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz
"""

from parameters import *
from functions import *
from sympy import LeviCivita
import autograd.numpy as np
from autograd import grad, jacobian
from scipy import optimize

# =============================================================================
# From Gralla et. al. https://iopscience.iop.org/article/10.3847/1538-4357/833/2/258/pdf
# Magnetic field relationships
# =============================================================================
# Make the 4D levi civita tensor
LeviCivita_tensor = np.zeros((4, 4, 4, 4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                LeviCivita_tensor[i, j, k, l] = int(LeviCivita(i, j, k, l)) # As defined in flatspace det(g) = 1
                
def Radial_eigenfunction_1(r):
    "Returns the first radial eigenfunction"
    r_over_r = r_s_val / r
    f = 1. - r_over_r
    return -1. * (3. / (2.*r)) * (3. - 4.*f + np.power(f, 2) + 2.* np.log(f)) / np.power(r_over_r, 3)

def mag_alpha(t, r, theta, phi):
    "Eqn (16) of Gralla 2019"
    mu = B_1 * R_ns ** 2  * Radial_eigenfunction_1(r) / Radial_eigenfunction_1_R_ns
    return mu * np.power(np.sin(theta), 2)

def dalphadtheta(t, r, theta, phi):
    "partial derivative of Eqn (16) of Gralla 2019 w.r.t. theta"
    mu = B_1 * R_ns ** 2  * Radial_eigenfunction_1(r) / Radial_eigenfunction_1_R_ns
    return mu * 2. * np.sin(theta) * np.cos(theta)

def mag_beta(t, r, theta, phi):
    """Eqn (16) of Gralla 2019.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    Returns:
        phi - Rot_ns * t
    """
    return phi - Rot_ns * t

B_1 = B_0 / 2. # Below equation (66) of Gralla et. al. 2016
Radial_eigenfunction_1_R_ns = Radial_eigenfunction_1(R_ns) # Define value of function as NS surface

# When aligned, some derivatives simplify or aren't necessary
dalphadt = 0. # partial alpha / partial t
dalphadr = grad(mag_alpha, argnum=1) # partial alpha / partial r
dalphadphi = 0. # partial alpha / partial phi

dbetadt = - Rot_ns #partial beta / partial t
dbetadr = 0. # partial beta / partial r
dbetadtheta = 0. #partial beta / partial theta
dbetadphi = 1. #partial beta / partial phi

def Number_Density(t, r, theta, phi):
    """Using the equations from https://iopscience.iop.org/article/10.3847/1538-4357/833/2/258/pdf
    which eventually return the number density of charged particles in the
    magnetosphere.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    Returns:
        num_density = the number density of charged particles around the NS
    """
    # Variables present in multiple terms.
    omega = Rot_ns
    I_hat = 2. / 5. * M_ns * R_ns ** 2 # Assuming uniform solid sphere
    omega_z = 2. * I_hat * G / (r**3) * omega # eqn(2) frame drag frequency (times by G to fix units)
    f = 1. - r_s_val / r
    
    # When aligned, these terms simplify to zero.
    # first_term = 0
    # second_term = 0
    # third_term = 0
    
    # Non-zero terms
    f4 = lambda t, r, theta, phi: np.power(r, 2) * dalphadr(t, r, theta, phi)
    f5 = lambda t, r, theta, phi: np.sin(theta) * dalphadtheta(t, r, theta, phi)
    fourth_term = dbetadphi * (f * grad(f4, argnum=1)(t, r, theta, phi) + grad(f5, argnum=2)(t, r, theta, phi) / np.sin(theta))
    
    # From eqn. (4a)
    J_up_t = -1. * (omega - omega_z) / (np.power(f, 1/2) * np.power(r, 2)) * (fourth_term)

    # From the result below eqn. (12)
    num_density = J_up_t / q_e # Charge density divided by charge
    
    return num_density

def electromagnetic_tensor(t, r, theta, phi):
    """Returns the electromagnetic tensor using the relation 
    F_uv = d_u alpha d_v beta - d_v alpha d_u beta
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    Returns:
        F_ab = The electromangetic tensor at the given point.
    """
    # Collecting the partial derivatives into vectors
    duda = np.array([dalphadt, dalphadr(t, r, theta, phi), dalphadtheta(t, r, theta, phi), dalphadphi])
    dudb = np.array([dbetadt, dbetadr, dbetadtheta, dbetadphi])
    
    # Taking the sum of the previously made vectors
    F_ab = np.einsum('i,j->ij', duda, dudb) - np.einsum('i,j->ji', duda, dudb)
            
    return F_ab

def Magnetic_Field(t, r, theta, phi):
    """Return the magnetic field vector at (r, theta, phi) and time t
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    returns:
        A 4 vector containing the components of the magnetic field at the given
        point in space. First (0th) component is zero.
    """
    
    f = 1. - r_s_val / r #Schwarzschild function
    U = np.array([np.sqrt(f), 0, 0, 0]) # Global velocity in the pulsars rotating frame U_a
    
    F_ab = electromagnetic_tensor(t, r, theta, phi) # Electromagnetic tensor
    
    # Det of metric due to not being in a flatspace time on the Levi-Civita tensor with raised indicies https://en.wikipedia.org/wiki/Levi-Civita_symbol Sec: Levi-Civita tensors
    B = 1/2 * np.einsum('ijkl,ij,k', LeviCivita_tensor, F_ab, U) * (-1. / np.sqrt(np.abs(np.linalg.det(metric(r, theta, phi)))))
    return B

def Omega_p_sq(t, r, theta, phi):
    """Compute the square of the plasma frequency. Take the absolute values as 
    number density can be both negative or positive, but plasma frequency 
    (and its square) is always positive.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        r_s = Schwarzschild radius
        
    Return:
        w_p_sq = The plasma fequency squared at the given point in space.
    """
    w_p_sq = np.abs(Number_Density(t, r, theta, phi) * q_e ** 2 / m_e)
    return w_p_sq

def Omega_p(t, r, theta, phi):
    """Compute the plasma frequency.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        r_s = Schwarzschild radius
        
    Return:
        w_p = The plasma fequency at the given point in space.
    """
    w_p = np.power(Omega_p_sq(t, r, theta, phi), 1/2)
    return w_p

# =============================================================================
# Plasma frequency partial derivatives. To be passed onto main.py
# =============================================================================
dw_p_dr = grad(Omega_p, argnum=1) #partial omega / partial r
dw_p_dtheta = grad(Omega_p, argnum=2) #partial omega / partial theta
dw_p_dphi = grad(Omega_p, argnum=3) #partial omega / partial phi

# =============================================================================
# Solution of the axion-photon conversion surface
# =============================================================================
def Conversion_Surface_scalar(theta, phi, t):    
    """
    Computes the radius at which photon axion conversion occurs 
    m_a/hbar = w_p when m_a is given in eV.
    
    Use for scalar inputs.
    
    Arguments:
        theta = polar angle from the NS norht pole
        phi = azimuthal angle from the positive x-axis
        t = current time
        
    Returns:
        r_conversion = the distance from the centre of the NS to the edge of 
                       the conversion surface at the given angles and time
    """
    f1 = lambda r: Omega_p_sq(t, r, theta, phi) - m_a**2 
    try:
        # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
        r_conversion = optimize.root_scalar(f1, bracket=[r_s_val *1.1, 110 * R_ns]).root
    except ValueError:
        r_conversion = None
    return r_conversion

def Conversion_Surface(theta_array, phi_array, t_array):
    """
    Computes the radius at which photon axion conversion occurs 
    m_a/hbar = w_p when m_a is given in eV. Using a root solve method
    
    Use for array inputs.
    
    Arguments:
        theta = polar angle from the NS norht pole
        phi = azimuthal angle from the positive x-axis
        t = current time
        
    Returns:
        r_conversion = the distance from the centre of the NS to the edge of 
                       the conversion surface at the given angles and time
    
    """
    try:
        r_conversion = np.zeros(theta_array.size)
        for i, theta in enumerate(theta_array):
            phi = phi_array[i]
            t = t_array[i]
            f1 = lambda r: Omega_p_sq(t, r * R_ns, theta, phi) - m_a**2
            try:
                # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
                r_conversion[i] = optimize.root_scalar(f1, bracket=[r_s_val / R_ns *1.1, 150]).root * R_ns # Find the r when w_p^2=m_a^2
            except ValueError:
                r_conversion[i] = None # Otherwise conversion does not occur
    # In the case that the inputs are not arrays and are instead floats
    except:
        f1 = lambda r: Omega_p_sq(t_array, r, theta_array, phi_array) - m_a**2 
        try:
            # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces
            r_conversion = optimize.root_scalar(f1, bracket=[r_s_val *1.1, 150 * R_ns]).root # Find the r when w_p^2=m_a^2
        except ValueError:
            r_conversion = None # Otherwise conversion does not occur
    return r_conversion

# =============================================================================
# Extra functions
# =============================================================================

def magnetic_field_unit_vec(t, r, theta, phi):
    """Returns the magnetic field 4-unit vector in the assoicated metric space-time
    
    Arguments:
        t = time
        r = radius
        theta = polar angle 
        phi = azimuthal angle
        
    Returns:
        b = the unit vector pointing in the direction of the magnetic field
    """
    B = Magnetic_Field(t, r, theta, phi) # (B^t, B^r, B^theta, B^phi)
    
    g_ab = metric(r, theta, phi) # Metric
    
    B_sq = np.matmul(np.matmul(g_ab, B), B) # Magnitude of the spatial part of the magnetic field (B_u*B^u)
    
    b =  B / np.sqrt(B_sq) # hat(b)^a
    return b

# =============================================================================
# Partial derivaties for the ray tracing equations. To be passed onto main.py
# =============================================================================

dw_p_sq_dt = grad(Omega_p_sq, argnum=0) #partial omega / partial t
dw_p_sq_dr = grad(Omega_p_sq, argnum=1) #partial omega / partial r
dw_p_sq_dtheta = grad(Omega_p_sq, argnum=2) #partial omega / partial theta
dw_p_sq_dphi = grad(Omega_p_sq, argnum=3) #partial omega / partial phi

db_dt = jacobian(magnetic_field_unit_vec, argnum=0) #partial b / partial t
db_dr = jacobian(magnetic_field_unit_vec, argnum=1) #partial b / partial r
db_dtheta = jacobian(magnetic_field_unit_vec, argnum=2) #partial b / partial theta
db_dphi = jacobian(magnetic_field_unit_vec, argnum=3) #partial b / partial phi

