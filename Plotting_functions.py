# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:21:00 2024

@author: jsa113

The functions in here are used to supplement the calculate_power.py code.

Using matplotlib to make the detector plane images.
"""
from parameters import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy import add


def plot_conversion_image(indicies, dPdO_array, max_x_R_ns, total_res, Obs_theta, time, Obs_phi=0, t=0, repeats=False):
    """
    Plots the image recieved by the detector plane
    """
    
    fig, ax = plt.subplots(dpi=500)
    max_x = max_x_R_ns * R_ns
    fine_step = 2. * max_x / total_res # Distance between adjacent fine pixels
    
    fine_x_array = np.linspace(-max_x, max_x, total_res, endpoint=False) + fine_step / 2.
    fine_y_array = np.linspace(-max_x, max_x, total_res, endpoint=False) + fine_step / 2.
    
    fine_x_mesh, fine_y_mesh = np.meshgrid(fine_x_array, fine_y_array)
    
    Z = np.zeros(fine_x_mesh.shape)
    
    if repeats:
        row_indicies, col_indicies = indicies[:,0], indicies[:,1]
        add.at(Z, (row_indicies, col_indicies), dPdO_array) # Comulative sum on each of the repeated indicies
        
    else:
        unique_indicies, mask = np.unique(indicies, return_index=True, axis=0)
        
        row_indicies, col_indicies = unique_indicies[:,0], unique_indicies[:,1]
        Z[row_indicies, col_indicies] = dPdO_array[mask]
        
    if (Z != 0).size != 0:
        Z_max_power = np.ceil(np.log10(np.max(Z)))
        Z_min_power = np.floor(np.log10(np.min(Z[Z != 0])))
    else:
        Z_max_power = 1
        Z_min_power = 0
    
    pcol1 = ax.pcolormesh(fine_x_mesh / R_ns, -fine_y_mesh / R_ns, Z.T, 
                           cmap='viridis', 
                           shading='nearest', 
                           norm=matplotlib.colors.LogNorm(vmin=10 ** Z_min_power, vmax=10 ** Z_max_power))
    
    ax.set_xticks(ticks=np.linspace(-max_x_R_ns, max_x_R_ns, 21), minor=True)
    ax.set_yticks(ticks=np.linspace(-max_x_R_ns, max_x_R_ns, 21), minor=True)
    ax.set_xticks(ticks=np.linspace(-max_x_R_ns, max_x_R_ns, 11))
    ax.set_yticks(ticks=np.linspace(-max_x_R_ns, max_x_R_ns, 11))
    ax.tick_params(top=True, right=True, direction='in', which='both')
    ax.set_aspect('equal')
    
    ax.set_xlabel(r'$\frac{x}{R_{ns}}$')
    ax.set_ylabel(r'$\frac{y}{R_{ns}}$', rotation=0)
    
    fig.colorbar(pcol1, orientation='vertical', label = r'$\frac{dP}{d\Omega} (W\mathrm{sr}^{-1}$)')
        
    # ax.set_title(("Mag_field = {}, Dispersion = {}\n"
    #                 "Obs_theta = {}, Obs_phi = {}, t_0 = {},\n"
    #                 "total_resolution = {}, probability = {}"
    #                 ).format(mag_field_choice, dispersion_method, int(Obs_theta * 180 / np.pi), int(Obs_phi  * 180 / np.pi), phase, total_resolution, probability_method))
    ax.set_title('Obs_theta = {}, Obs_phi = {}, t = {}'.format(Obs_theta, Obs_phi, time))
    plt.show()
    return Z