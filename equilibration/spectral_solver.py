import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from os import system, name
from scipy.optimize import bisect
import pandas as pd
import matplotlib.pyplot as plt
from spectral_state import State
from astropy.constants import c

class Solver:
    c = 29979245800.0
    HBARC = 1.97327e-11 #MeV * cm

    def __init__(self, table : Table, state : State, opacity_flags : dict, opacity_pars : dict, num_bins : float, nu_external_E : list):
        self.table = table
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        self.num_bins = num_bins
        self.nu_external_E = nu_external_E #[nue_n_dist, anue_n_dist]
        self.E_dist = None


    def temperature_from_e(self, state : State) -> None:
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                state [State]: the state object containing the current simulation conditions.

            Modifies:
                state.t [float]: the calculated temperature [MeV]
        """

        log_e = np.log((self.table.thermo["Q7"] + 1) * self.table.mn * state.nb * 1e-39)
        # Find difference between calculated e and actual e
        def f(t):
            e_interp = self.table.eval_given_rtx(log_e, np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([t]), method="linear")[0]
            return (state.fluid_e * 1e-39) - np.exp(e_interp)

        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")

        state.t = t
        return
    
    def get_potentials(self, state : State) -> None:
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Inputs:
                state [State]: the state object containing the current simulation conditions

            Modifies:
                state.mu_p [float]: the state's proton chemical potential [MeV]
                state.mu_n [float]: the state's neutron chemical potential [MeV]
                state.mu_e [float]: the state's electron chemical potential [MeV]
        """


        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn 
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q  #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        state.mu_p = mu_p
        state.mu_n = mu_n
        state.mu_e = mu_e
        return 

    def calculate_corrector_quantities(self, state : State) -> None:
        """
            Calulates quantities needed for BNS_NURATES corrections that are relevant at high density. 

            Inputs:
                state [State]: the state object containing the current simulation conditions

            Modifies:
                state.mp_eff [float]: Proton Dirac effective mass [MeV]
                state.mn_eff [float]: Neutron Dirac effective mass [MeV]
                state.dm_eff [float]: The effective mass difference between neutrons and protons [MeV]
        """

        #Use eos.qK to find dirac effective masses to calculate dm_eff, don't attempt dU for now

        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff

        state.mn_eff = mn_eff
        state.mp_eff = mp_eff
        state.dm_eff = dm_eff
        return 

    def fermi_dirac_n(self, energy : float, mu : float, t : float) -> float:
        """
            For a given temperature, chemical potential, and energy, calculates the number density of neutrinos according to the Fermi-Dirac distribution.

            Inputs:
                energy: neutrino energy [MeV]
                mu:     chemical potential [MeV]
                t:      temperature [MeV]
            
            Outputs:
                [float]: the neutrino number density for the specified energy, chemical potential, and temperature. Each bin is in units of cm^-3
        """
        return ((1 / (np.exp((energy - mu) / t) + 1))) * ((energy ** 2) / (2 * np.pi ** 2 )) * (1 / (self.HBARC ** 3)) #needs to be divided by HBARC^3 to be in units of cm^-3

    def calculate_dist(self, state : State) -> None:
        """
            Calculates the energy and number density distributions across the electron neutrinos and electron antineutrinos. This function assumes the temperature, potentials, effective masses,
            and effective mass differences have already been calculated for the current state.

            Inputs:
                state [State]: the state object containing the current simulation conditions
            
            Modifies:
                state.nu_n_dist [List]: A list of numpy arrays containing the binned distribution of neutrino number densities per species [cm^-3]. The species are ordered [nue, anue].

        """

        #TODO: Needs error handling if temperature, potentials, effective masses, or effective mass differences haven't been calculated yet.

        #Calculate nue and anue chemical potentials
        mu_nue = state.mu_p - state.mu_n + state.mu_e
        mu_anue = - mu_nue

        #construct energy distribution with logarithmic spacing
        E_dist_edges = np.exp(np.linspace(np.log(10 ** -2), np.log(10 ** 4), self.num_bins + 1))
        E_binwidths = np.diff(E_dist_edges)
        self.E_dist = E_dist_edges[:-1] + (0.5 * E_binwidths)

        #calculate energy and number density distributions using fermi-dirac
        nue_n_dist = self.fermi_dirac_n(self.E_dist, mu_nue, state.t) * E_binwidths
        anue_n_dist = self.fermi_dirac_n(self.E_dist, mu_anue, state.t) * E_binwidths

        nue_J_dist = nue_n_dist * self.E_dist
        anue_J_dist = anue_n_dist * self.E_dist

        state.nu_n_dist = [nue_n_dist, anue_n_dist]
        state.nu_J_dist = [nue_J_dist, anue_J_dist]

        return

    
    def calculate_spectral_rates(self, state : State) -> None:
        """
            Calculates the neutrino-matter reaction rates for each energy bin in the neutrino distributions. 

            Inputs:
                state [State]: the state object containing the current simulation conditions.
            
            Modifies:
                state.rates [List]: A list of dictionaries, where each dictionary contains the species-specific reaction rates for a particular energy bin of the neutrino distributions.
        """
        
        #Calculate total number and energy densities by species for bns_nurates
        state.n_m1 = [np.sum(state.nu_n_dist[0]), np.sum(state.nu_n_dist[1]), 0, 0] #nue, anue, nux, anux
        state.J_m1 = [np.sum(state.nu_J_dist[0]), np.sum(state.nu_J_dist[1]), 0, 0] #nue, anue, nux, anux

        #Initialize bns_nurates eos_pars and populate
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = state.nb * 1e-21
        eos_pars.temp = state.t
        eos_pars.ye = state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = state.mu_n
        eos_pars.mu_e = state.mu_e
        eos_pars.mu_p = state.mu_p
        eos_pars.dm_eff = state.dm_eff

        #Set up Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 9
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars

        #Load non-distribution m1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = state.chi_m1

        #Load grey pars
        grey_pars = bns.GreyOpacityParams()
        grey_pars.eos_pars = eos_pars
        grey_pars.opacity_flags = opacity_flags

        rates = []

        #Calculate rates; one set of rates for each energy in E_dist
        for i in range(0, len(self.E_dist)):
            m1_pars.n = [x * 1e-21 for x in state.n_m1]
            m1_pars.J = [x * 1e-21 for x in state.J_m1]

            distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)
            grey_pars.distr_pars = distr_pars
            grey_pars.m1_pars = m1_pars

            spectral_rates = bns.ComputeSpectralOpacitiesStimulatedAbs(self.E_dist[i], quad, grey_pars)
            spectral_rates['kappa'] = [x * 1e7 for x in spectral_rates['kappa']]
            spectral_rates['kappa_s'] = [x * 1e7 for x in spectral_rates['kappa_s']]
            rates.append(spectral_rates)

        state.rates = rates
        return

    def step(self, state: State, timestep : float, e_ratio_range : list, ye_ratio_range : list):
        """
            An implicit numerical solver that determines the next state of the simulation. Given limits on the change in electron fraction and baryon energy density and a timestep guess, determines
            an appropriate timestep and updates the state with the fluid's properties after the timestep.

            Inputs:
                state [State]: the state object containing the current simulation state.
                timestep [float]: the simulation timestep [s]. Dynamically changes depending on how quickly the fluid energy density and electron fraction change to preserve stability
                e_ratio_range [list]: a list containing the desired range dictating the acceptable change in fluid energy density, expressed as a ratio normalized around 1
                ye_ratio_range [list]: a list containing the desired range dictating the acceptable change in electron fraction, expressed as a ratio normalized around 1
            
            Modifies:
                state.fluid_e [float]: the fluid's baryon energy density [MeV cm^-3]
                state.ye [float]: the fraction of electrons to the total number of baryons present in the fluid
                state.nu_n_dist [list]: a list containing the per-species neutrino number density distributions, ordered such that [nue, anue]
                state.time [float]: the current simulation time [s]
        """
        
        #Calculate number density distribution of incoming neutrinos
        nue_ext_n = self.nu_external_E[0] / self.E_dist 
        anue_ext_n = self.nu_external_E[1] / self.E_dist 

        E_before = np.sum(state.J_m1[0]) + np.sum(state.J_m1[1]) + state.fluid_e + np.sum(self.nu_external_E[0]) + np.sum(self.nu_external_E[1]) #fluid energy density + total neutrino number density
        lep_num_before = (state.nb * state.ye) + (np.sum(state.n_m1[0]) - np.sum(state.n_m1[1])) + (np.sum(nue_ext_n) - np.sum(anue_ext_n)) #electron density + net neutrino density

        #calculate kappa for each species per energy bin, converting to inverse seconds
        kappa_nue = np.array([kappa * c.to("cm/s") for kappa in state.rates['kappa'][0]])
        kappa_anue = np.array([kappa * c.to("cm/s") for kappa in state.rates['kappa'][1]])


        #Evolve system and check for differences in fluid energy and electron fraction. Recompute with altered timestep if the conditions get too stiff.
        acceptable_timestep = False
        while not acceptable_timestep:
            #Calculate new number density distributions for the two neutrino species
            new_nue_n_dist = ((np.array(state.nu_n_dist[0]) + ((kappa_nue * timestep) * nue_ext_n)) / (1 + (kappa_nue * timestep)))
            new_anue_n_dist = ((np.array(state.nu_n_dist[1]) + ((kappa_anue * timestep) * anue_ext_n)) / (1 + (kappa_anue * timestep)))

            #Find baryon energy density after timestep
            new_e_fluid = -np.sum(((new_nue_n_dist - state.nu_n_dist[0]) + (new_anue_n_dist - state.nu_n_dist[1])) * self.E_dist) + state.fluid_e

            #Find electron fraction after the timestep
            new_ye = ((1 / state.nb) * np.sum((new_anue_n_dist - state.nu_n_dist[1]) - (new_nue_n_dist - state.nu_n_dist[0]))) + state.ye

            #Calculate new total energy density and lepton number density
            E_after = np.sum(new_nue_n_dist * self.E_dist) + np.sum(new_anue_n_dist * self.E_dist) + new_e_fluid + np.sum(nue_ext_n) + np.sum(anue_ext_n)
            lep_num_after = (state.nb * new_ye) + np.sum(new_nue_n_dist) + np.sum(new_anue_n_dist) + np.sum(nue_ext_n) + np.sum(anue_ext_n)

            #Calculate energy density and lepton number error
            E_error = E_after - E_before
            lep_num_error = lep_num_after - lep_num_before

            #Feed errors back into fluid to enforce conservation laws
            new_e_fluid += E_error
            ne_ye += (lep_num_error / state.nb)

            #Calculate ratios of change for matter fields, normalize by 1 
            e_ratio = 1 - (new_e_fluid - state.fluid_e)
            ye_ratio = 1 - (new_ye - state.ye)

            #Double or halve timestep as needed. Small changes allow for larger timesteps, whereas large changes should be constrained by smaller timestep values.
            if e_ratio < e_ratio_range[0] or ye_ratio < ye_ratio_range[0]:
                timestep *= 2

            elif e_ratio > e_ratio_range[1] or ye_ratio > ye_ratio_range[1]:
                timestep /= 2

            else:
                acceptable_timestep = True

        #Overwrite the current state's variables with the evolved ones
        state.ye = new_ye
        state.fluid_e = new_e_fluid
        state.nu_n_dist = [new_nue_n_dist, new_anue_n_dist]
        state.time += timestep
        return       
