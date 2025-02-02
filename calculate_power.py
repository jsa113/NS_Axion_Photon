# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:42:27 2023

@author: Jesse Satherley

This files purpose is to plot the timeseries radiated power. It does this by 
taking already produced data and ploting it as a time series.

Use the simulation parameters to decide on what data is being imported.

Use the date_str_list to decide on what exact simulations are being imported.
"""

from parameters import * # The file containing all the simulation parameters
from functions import * # The file containing all the extra functions
from Dispersion_methods import *
from Plotting_functions import *
import os
import numpy as np
from numpy import char
import matplotlib.pyplot as plt
import matplotlib
import imageio
from tqdm import tqdm

# =============================================================================
# Choices for the power calculation and plots
# =============================================================================

P_ayy_is_1 = False

repeats = True
x_axis_plot = 'polar_angle' # Choose from 'polar_angle' or 'phase'
Probability_method_choice = 'iso' # Choose from '1D', 'iso'
Plot_conversion_surface = False
produce_gif = False
include_MW_data = False

use_saved_power = True

# =============================================================================
# File name of the data
# 
# Use .extend('[]') to add the date of the simulation for the file paths
# e.g. date_str_list.extend(['2025_01_01'])
#
# Use mag_field_choices to describe the magnetic field used by a simulation
#
# Use dispersion_method_choices to describe the dispersion method used by a simulation
#
# These are list and each element in each list conrespondes to a simulation.
# You can put multiple simulations in to compare the power.
# =============================================================================
date_str_list = []

date_str_list.extend(['2025_02_02_(2)'])

mag_field_choices = ['GJ']
dispersion_method_choices = ['unmagnetised']

# =============================================================================
# Code to plot the time series
# =============================================================================

# Empty lists to fill with the data from each individual simulation set
total_radiated_power_list = list()
times_list = list()
polar_angles_list = list()
frames_list = [] # Used for rotating gif of the detector image
sim_dir_list = [] # Empty list for the file path to each simulation's results

# Picks the assoicated array with a the probability method.
Probability_dict = {
    '1D': 0,
    'iso': 1,
    'millar': 2,
    'aniso': 3}
Probability_method = Probability_dict[Probability_method_choice]

def calculate_power(photon_conversion_data, conversion_indicies, sim_paraters, repeats=True):
    """
    
    """
    max_x_R_ns, coarse_resolution, fine_resolution, m_a, g_ayy, DM_pho_inf, M_ns = sim_paraters
    
    if not repeats: # Can be used to only allow a photon to convert once
        _, mask = np.unique(conversion_indicies, return_index=True, axis=0) # A mask that finds the first instance of a photon
        conversion_indicies = conversion_indicies[mask] 
        photon_conversion_data = photon_conversion_data[mask] # Removes multiple intercepts from a simulation
    
    num_of_photons = photon_conversion_data.shape[0] # Number of converting photons in a simulation
    
    # Empty arrays to be filled with the fine search results
    theta_k_array = np.zeros(num_of_photons)
    
    dPdO_array_1D = np.zeros(num_of_photons)
    dPdO_array_iso = np.zeros(num_of_photons)
    dPdO_array_prob_is_1 = np.zeros(num_of_photons)
    
    conversion_p_1D_array = np.zeros(num_of_photons)
    conversion_p_iso_array = np.zeros(num_of_photons)

    # =============================================================================
    # Find the conversion probability and the radiated power for each pixel
    # ============================================================================= 
    for i in tqdm(range(num_of_photons)): #iterate over each converted photon
    
        photon_conversion = photon_conversion_data[i] # Grab the ith converting photon
        
        r_s_val = 2. * G * M_ns # Schwarzschild radius
        
        g_ab = metric(photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val) # Metric at the point of conversion
        inv_g_ab = inv_metric(photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val) # Inverse of the metric at the point of conversion
        
        B = Magnetic_Field(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val) # Magnetic field vector
        k_a = np.array([photon_conversion[4], photon_conversion[5], photon_conversion[6], photon_conversion[7]]) # Covariant Photon 4-momentum
        k_up_a = np.matmul(inv_g_ab, k_a) # Contravariant photon 4-momentum
        
        k_spac = k_up_a[1:] # The spatial component of the contravariant photon 4-momentum
        
        k_sq = -1 * np.einsum('ij,i,j', inv_g_ab, k_a, k_a) # The magnitude of the 4-momentum
        k_spac_norm = np.sqrt(np.einsum('ij,i,j', g_ab[1:, 1:], k_spac, k_spac)) # The magnitude of the 3-momentum
        k_spac_unit = k_spac / k_spac_norm # The 3-momentum unit vector
        B_sq = np.einsum('ij,i,j', g_ab, B, B) # The magnitude of the magnetic field squared
        
        # Find the angle between the magnetic field and the photon
        dot_product = np.dot(B, k_a) / (np.sqrt(B_sq) * k_spac_norm)
        theta_k = np.arccos(dot_product)
        theta_k_array[i] = theta_k # Save to an array for comparison 
        
        # The partial derivative of the plasma frequency in each spatial direction
        grad_w_p = np.array([dw_p_dr(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val),
                              dw_p_dtheta(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val),
                              dw_p_dphi(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3])], r_s_val)

        # Velocity of axion at the point of conversion
        v_em = np.sqrt(2* G* M_ns / photon_conversion[1])
        # Plasma frequency squared
        w_p_sq = Omega_p_sq(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val)
        
        # =============================================================================
        # Find the 1D conversion_probability
        # =============================================================================
       
        grad_w_p_em = np.dot(k_spac_unit, grad_w_p) # Plasma gradient in the direction of the photon
        
        conversion_p_1D = np.abs(np.pi * (g_ayy / 1e9) ** 2 * B_sq * np.sin(theta_k) ** 2 / (2 * grad_w_p_em * v_em)) # From 1D
        conversion_p_1D_array[i] = conversion_p_1D # Save to an array for comparison 

        U = Global_velocity_vector(photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val)
        W = U[0] * k_a[0] # k_a U^a
        
        refractive_index = np.power(1 + k_sq / W ** 2, 1/2)
        density_DM = 2 / (np.sqrt(np.pi) * DM_v_0) * np.sqrt(2 * G * M_ns / photon_conversion[1]) * DM_pho_inf
        
        dPdO_array_1D[i] = (fine_step / refractive_index) ** 2 * np.power(1 - r_s_val / photon_conversion[1], 3/2) * density_DM * v_em * conversion_p_1D / (4 * np.pi)
        dPdO_array_1D[i] = dPdO_array_1D[i] / Watts_to_eV_sq # Account for reflexions and convert to SI units
        
        # =============================================================================
        # Find the Isotropic conversion probability from MW23
        # =============================================================================
        E_gamma = np.sqrt(2* G* M_ns / photon_conversion[1] + 1) * m_a # Energy of emitted photon via energy conversion (E^2 = p^2 + m^2)
        
        # The partial derivative of the plasma frequency squared in each coordinate
        grad_w_p_sq = np.array([dw_p_sq_dt(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val),
                                dw_p_sq_dr(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val),
                                dw_p_sq_dtheta(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3], r_s_val),
                                dw_p_sq_dphi(photon_conversion[0], photon_conversion[1], photon_conversion[2], photon_conversion[3])], r_s_val)
        
        # Conversion probability for an isotropic plasma
        conversion_p_iso = pi * (g_ayy / 1e9) ** 2 * B_sq * np.sin(theta_k) ** 2 * E_gamma ** 2 / np.abs(np.dot(k_up_a, grad_w_p_sq))
        conversion_p_iso_array[i] = conversion_p_iso # Save to an array for comparison 
        
        k_0 = m_a * DM_v_0
        
        wavenumber_sq = k_spac_norm ** 2
        
        f_inf = (DM_pho_inf / m_a) / (4. * pi * k_0**2)
        n_c = (DM_pho_inf / m_a) * np.sqrt((k_0**2 + 2. * G * M_ns * m_a ** 2/ photon_conversion[1])) / k_0 
        
        f_c = v_em * n_c / (4. * pi * wavenumber_sq)
        
        # Power emitted for a converting photon in isotropic
        dPdO_array_iso[i] = fine_step ** 2 * E_gamma ** 3 * conversion_p_iso * f_c 
        dPdO_array_iso[i] = dPdO_array_iso[i]  / Watts_to_eV_sq # Convert from natural units to SI units
        
        
        # =============================================================================
        # If the probability is 1, the power is given as this.
        # =============================================================================
        dPdO_array_prob_is_1[i] = fine_step ** 2 * E_gamma ** 3 * f_c # Same as the above power, but excludes the conversion_p_iso
        dPdO_array_prob_is_1[i] = dPdO_array_prob_is_1[i] / Watts_to_eV_sq # Convert from natural units to SI units
        
    total_powers_arrays = [dPdO_array_1D, dPdO_array_iso, dPdO_array_millar, dPdO_array_aniso]
    np.save(folder + '/total_powers_arrays.npy', total_powers_arrays)
    return total_powers_arrays

for n, date_str in enumerate(date_str_list):  # Iterate over each simulation
    
    # Pick the corresponding magnetic field and dispersion relation for simulation
    try:
        mag_field_choice = mag_field_choices[n]
        dispersion_method = dispersion_method_choices[n]
        print('Magnetic field choice is: {}'.format(mag_field_choice))
        print('Dispersion Method choice is: {}'.format(dispersion_method))
    except:
        print('===== Choices should be unchanged =====')
        print('Magnetic field choice is: {}'.format(mag_field_choice))
        print('Dispersion Method choice is: {}'.format(dispersion_method))
        
    # Location of the data
    directory = './data/{}_magnetic_field/{}/'.format(mag_field_choice, dispersion_method) 

    # Search the folders for the data that is present
    folders = [x[0] for x in os.walk(directory)][1:]
    
    # Empty arrays to be filled
    total_radiated_power_array = np.array([])
    total_converted_axions_array = np.array([])
    times = np.array([])
    polar_angles = np.array([])
    frames = [] # Used for rotating gif of the detector
    
    simulation_folders = np.array(folders)[char.find(folders, date_str + "\\") != -1] # Filter to only look at a specific angle
    
    if simulation_folders.size == 0: # Skips over any simulations that has no data
        print('===== Simulation data missing! =====\nSkipping {}'.format(date_str))
        sim_dir_list.append(None)
        
    for folder in simulation_folders: # Iterates through a simulations data folders
        print(folder)
        sim_dir = folder[:folder.index(date_str)] + date_str
        
        if sim_dir not in sim_dir_list: # Adds the simulation to a list for later use
            sim_dir_list.append(sim_dir)
        
        # Load the parameter file to read off important simulation parameters
        with open(sim_dir + '/parameter_set.txt') as f:
            lines = f.readlines()

        # Import the appropriate magentic field functions
        mag_field_choice = np.array(lines)[char.find(lines, 'mag_field_choice') != -1][0].split()[-1]
        if mag_field_choice == 'GR':
            from GR_magnetic_field import *
        elif mag_field_choice == 'GJ':
            from GJ_magnetic_field import *

        # Initalise necessary variables
        max_x_R_ns = float(np.array(lines)[char.find(lines, 'max_x') != -1][0].split()[-2])
        coarse_resolution = int(np.array(lines)[char.find(lines, 'coarse_resolution') != -1][0].split()[-1])
        fine_resolution = int(np.array(lines)[char.find(lines, 'fine_resolution') != -1][0].split()[-1])
        m_a = float(np.array(lines)[char.find(lines, 'm_a') != -1][0].split()[-1])
        g_ayy = float(np.array(lines)[char.find(lines, 'g_ayy') != -1][0].split()[-1])
        DM_pho_inf = float(np.array(lines)[char.find(lines, 'DM_pho_inf') != -1][0].split()[-1]) * kg_to_eV / (m_to_per_eV ** 3) # Convert to eV
        M_ns_ratio = float(np.array(lines)[char.find(lines, 'M_ns') != -1][0].split()[-1])
        M_ns = M_ns_ratio * M_sol
        
        # Save as a list to be passed onto functions
        sim_paraters = [max_x_R_ns, coarse_resolution, fine_resolution, m_a, g_ayy, DM_pho_inf, M_ns]

        # More necessary variables
        total_resolution = coarse_resolution * fine_resolution
        max_x = max_x_R_ns * R_ns
        fine_step = 2 * max_x / total_resolution
        fine_x_array = np.arange(-max_x, max_x, fine_step) + fine_step / 2 
        fine_y_array = np.arange(-max_x, max_x, fine_step) + fine_step / 2
        fine_x_mesh, fine_y_mesh = np.meshgrid(fine_x_array, fine_y_array)
        
        # Check if the conversion data already exists for a simulation
        try:
            photon_conversion_data = np.load(folder + '/conversion_point.npy')
            photon_conversion_data = photon_conversion_data.reshape((int(photon_conversion_data.size/8), 8))
        except:
            # Load ray traced photon data to get the values at the conversion surface
            photon_t_data = np.load(folder + '/photon_data_t.npy', allow_pickle=True)
            photon_data_lens = np.array([len(i) for i in photon_t_data])
            num_of_photons = photon_t_data.size
            if photon_t_data.size == 0: # Skips over any simulations that returned zero converted photons
                photon_conversion_data = np.zeros(8)
                
            else:
                photon_data_cumlens = np.cumsum(photon_data_lens) # Cumulative lengths of each data
                
                photon_conversion_data = np.zeros((photon_t_data.size, 8))
                photon_conversion_data[:, 0] = np.hstack(photon_t_data)[photon_data_cumlens - 1]
                
                data_str_list = ['/photon_data_r.npy', '/photon_data_theta.npy', '/photon_data_phi.npy', '/photon_data_w.npy', '/photon_data_kr.npy', '/photon_data_ktheta.npy', '/photon_data_kphi.npy']
                for n, data_str in enumerate(data_str_list):
                    photon_data = np.load(folder + data_str, allow_pickle=True)
                    photon_conversion_data[:, n + 1] = np.hstack(photon_data)[photon_data_cumlens - 1]
                
        conversion_indicies = np.load(folder + '/conversion_index.npy') # Load the data
        conversion_indicies = conversion_indicies.reshape((int(conversion_indicies.size/2), 2)).astype(int) # Fix the data structure to be more convinent
        
        # Check if the power data already exists for a simulation
        if use_saved_power:
            try:
                total_powers_arrays = np.load(folder + '/total_powers_arrays.npy')
            except:
                print("===== Total Powers array file is missing! =====\nCalculating power instead")
                total_powers_arrays = calculate_power(photon_conversion_data, conversion_indicies, sim_paraters, repeats)
        
        else:
            total_powers_arrays = calculate_power(photon_conversion_data, conversion_indicies, sim_paraters, repeats)
            
        time_index = folder.index('t_')
        time = float(folder[time_index + 2:]) # Extract time
        times = np.append(times, time) # Append to time data array
        
        polar_angle_index = folder.index('Obs_theta_')
        polar_angle = float(folder[polar_angle_index + 10:time_index - 1]) # Extract time
        polar_angles = np.append(polar_angles, polar_angle) # Append to time data array
        
        # Store total power for simulation in an array
        total_powers = np.sum(total_powers_arrays, axis=1)
        try:
            total_radiated_power_array = np.vstack((total_radiated_power_array, total_powers))
        except: 
            total_radiated_power_array = total_powers
            
        if Plot_conversion_surface: # Used to plot an image of the detector plane
            try: # Checks if the conversion index file is present in the data
                total_radiated_power = plot_conversion_image(conversion_indicies, total_powers_arrays[Probability_method], max_x_R_ns, total_resolution, polar_angle, time, repeats=repeats)
            except:
                print('=' * 35  + '\nMissing conversion index file!\n' + '=' * 35)
                
        if produce_gif: # The following is used for rotating gif of the detector
            
            fig_frames, ax_frames = plt.subplots(figsize=(10, 10), dpi=500)
            pcol1 = ax_frames.imshow(total_radiated_power.T, 
                                      cmap='viridis',
                                      norm=matplotlib.colors.LogNorm(vmin=1e7, vmax=1e11))
            
            fig_frames.colorbar(pcol1, orientation='vertical', label = r'$\frac{dP}{d\Omega} (W\mathrm{sr}^{-1}$)')
            # plt.title("t = {}".format(time))
            fig_frames.suptitle(r"$\theta={}$".format(polar_angle))
            fig_frames.canvas.draw()
            
            mat = np.array(fig_frames.canvas.renderer._renderer)
            
            frames.append(mat)
            plt.close()
            
    total_radiated_power_list.append(total_radiated_power_array) # Append power array to simulation powers list
    times_list.append(times) # Append time array to all angles list
    polar_angles_list.append(polar_angles)
    frames_list.append(frames)
    
# Plot multiple simulations on the same plot
fig, ax = plt.subplots(dpi=250, figsize=(4,4)) # Create the plot to be used
line_styles = np.tile(['dashed', 'solid', 'dotted'], len(date_str_list) // 3 + 1) # Choose the alterating line styles for the different simulations plotted together

for n, date_str in enumerate(date_str_list): # Iterates through each simulation that is being plotted
    if times_list[n].size == 0: # Checks whether a simulation folder is empty and skips it
        None
    else:
        # Reads the individual simulation parameters from the parameter file
        with open(sim_dir_list[n] + '\parameter_set.txt') as f:
            lines = f.readlines()
            
        # Initalise necessary variables for plotting information
        m_a = float(np.array(lines)[char.find(lines, 'm_a') != -1][0].split()[-1])
        metric_choice = np.array(lines)[char.find(lines, 'metric_choice ') != -1][0].split()[-1]
        coarse_resolution = float(np.array(lines)[char.find(lines, 'coarse_resolution ') != -1][0].split()[-1])
        fine_resolution = float(np.array(lines)[char.find(lines, 'fine_resolution ') != -1][0].split()[-1])
        total_res = coarse_resolution * fine_resolution
        mag_field_choice = np.array(lines)[char.find(lines, 'mag_field_choice') != -1][0].split()[-1]
        dispersion_method = np.array(lines)[char.find(lines, 'dispersion_method') != -1][0].split()[-1]
        M_ns_ratio = float(np.array(lines)[char.find(lines, 'M_ns') != -1][0].split()[-1])

        if x_axis_plot == 'polar_angle': # Checks the x_axis is choosen to be the polar angle or the phases
            x = np.sort(polar_angles_list[n]) * pi / 180
            if P_ayy_is_1:
                try:
                    y = np.take_along_axis(total_radiated_power_list[n][:, -1], np.argsort(polar_angles_list[n]), axis=0) / 1.6e-19
                    
                except TypeError:
                    if total_radiated_power_list[n][-1].size == 1:
                        y = (total_radiated_power_list[n][-1]) / 1.6e-19
            else:
                if total_radiated_power_list[n].ndim == 1:
                    y = total_radiated_power_list[n][Probability_method] / 1.6e-19
                else:                    
                    y = np.take_along_axis(total_radiated_power_list[n][:, Probability_method], np.argsort(polar_angles_list[n]), axis=0) / 1.6e-19

            if dispersion_method == 'unmagnetised':
                plasma_type = 'isotropic'
            elif dispersion_method == 'magnetised':
                plasma_type = 'anisotropic'
            else:
                plasma_type = dispersion_method
                
            if mag_field_choice == 'GR':
                mag_field_choice = 'GLP'
            
            if include_MW_data: # Changes the legend if the MW24 data is to be plotted alongside
                ax.plot(x, y, marker='.', markersize=5, linestyle=line_styles[n], label=r'Our data $m_a={:.2f}\,\mu eV$'.format(m_a/1e-6))
                try:
                    MW_data = np.loadtxt('data/MW23_data/Isotropic_DiffPower_{}eV_.dat'.format(m_a))
                    x_data = MW_data[:,0]
                    y_data = MW_data[:,1]
                    ax.plot(x_data, y_data, marker='.', markersize=3, linestyle=line_styles[n], label=r'MW23 $m_a = {:.2f}\,\mu eV$'.format(m_a/1e-6), zorder=1)
                except:
                    print('===== Missing MW23 data! =====')
                
            else:
                ax.plot(x, y, marker='.', markersize=5, linestyle=line_styles[n], label='{}'.format(mag_field_choice))
                
        elif x_axis_plot == 'phase':
            Obs_theta = polar_angles_list[0][0]
            x = times_list[n]
            
            if P_ayy_is_1:
                y = total_radiated_power_list[n][:, -1]
            else:
                y = total_radiated_power_list[n][:, Probability_method]
               
            ax.plot(x, y, marker='.', markersize=5, linestyle=line_styles[n], label=r'${}^\degree$, res = {},  mag = {}'.format(Obs_theta, total_res, mag_field_choice))

        if produce_gif: # Makes the gif of the detector plane
            print("=== Producing .gif of the detector plane images ===")
            frames = np.append(np.array(frames_list[n])[np.argsort(polar_angles_list[n])][:-1], 
                                np.flip(np.array(frames_list[n])[np.argsort(polar_angles_list[n])], axis=0)[:-1], axis=0) # Takes the frames and does a reverse order so the gif cycles nicely, also removes the repeated frames
            imageio.mimsave('./plots/' + date_str + '.gif', frames, duration=200, loop=1000) # Joins all the frames and saves the result. Use duration to control the speed.
        
        
        # This loop adds the x and y values to the points on the graph
        # for i, x in enumerate(np.sort(polar_angles_list[n]) * pi / 180):
        #     y = (np.take_along_axis(total_radiated_power_list[n][:, -1], np.argsort(polar_angles_list[n]), axis=0) / 1.6e-19)[i]
        #     # first element
        #     if i == 0:
        #         plt.text(x+.2, y, f"({x:.3g}, {y:.3g} )", horizontalalignment="left")
        #     # last element
        #     elif i == len(np.sort(polar_angles_list[n]) * pi / 180) - 1:
        #         plt.text(x, y - 10, f"({x:.3g}, {y:.3g} )", horizontalalignment="right")
        #     else:
        #         plt.text(x, y-20, f"({x:.3g}, {y:.3g} )", horizontalalignment="left")
        

# =============================================================================
# Following code creates the legends and scales the plots
# =============================================================================
if x_axis_plot == 'polar_angle':
    ax.set_xlim(0.0, pi / 2)
    if P_ayy_is_1:
        ax.set_ylim(5e32, 5e35)
    else:
        # ax.set_ylim(5e29, 5e32)
        ax.set_ylim(1e30, 5e32)
        
    ax.set_yscale('log')
    ax.set_ylabel(r'$\langle\frac{dP}{d\theta_{obs}}\rangle$ (eV/s)')
    ax.set_xlabel(r'Observing angle, $\theta_{obs}$ (radians)')
    ax.legend()
    
    if include_MW_data:
        ax.legend(prop={'size': 6})
        ax.set_title(r'Isotropic $M_{{ns}}={:.1f}M_\odot$'.format(M_ns_ratio))
        plt.savefig('./plots/For_Paper/Isotropic_MW23.pdf', bbox_inches = "tight")
    else:
        ax.set_title(r'$m_a={:.2f}\,\mu eV$, $M_{{ns}}={:.1f}M_\odot$'.format(m_a/1e-6, M_ns_ratio))
        plt.savefig('./plots/For_Paper/Isotropic_ma_{:.2f}ueV_Mns_{}.pdf'.format(m_a/1e-6, M_ns_ratio), bbox_inches = "tight")
        
else: # If plotting phase instead
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1e39, 1e40)
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel(r'$\frac{dP}{d\Omega}$ (eV/s)')
    ax.set_xlabel(r'phase')
    
    if P_ayy_is_1:
        ax.set_title(r'Isotropic Plasma $P_{a\gamma\gamma}=1$, ' + r'$m_a={:.2f}\,\mu eV$'.format(m_a/1e-6))
    else:
        ax.set_title(r'Isotropic Plasma ' + r'$m_a={:.2f}\,\mu eV$'.format(m_a/1e-6))

plt.show()



