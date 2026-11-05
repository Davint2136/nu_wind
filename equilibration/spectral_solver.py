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
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        self.num_bins = num_bins
        self.nu_external_E = nu_external_E #[nue_n_dist, anue_n_dist]
        self.E_dist = None


    def temperature_from_e(self, state : State):
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
                nb [np.NDArray]: A 1D array consisting of one baryon number density in cm^-3.
                ye [np.NDArray]: A 1D array consisting of one charge fraction.
                e_val [float]: The fluid's energy density in MeV/cm^3.

            Outputs:
                t [float]: The fluid temperature in MeV.
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
        
        return t
    
    def get_potentials(self, state : State):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """


        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn 
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q  #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self, state : State):
        """
            Calulates quantities needed for BNS_NURATES corrections that are relevant at high density. 

            Outputs:
                mp_eff [float]: Proton Dirac effective mass [MeV]
                pn_eff [float]: Neutron Dirac effective mass [MeV]
                dm_eff [float]: The effective mass difference between neutrons and protons
        """

        #Use eos.qK to find dirac effective masses to calculate dm_eff, don't attempt dU for now

        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff
        return mp_eff, mn_eff, dm_eff

    def fermi_dirac_n(self, energy, mu, t):
        return ((1 / (np.exp((energy - mu) / t) + 1))) * ((energy ** 2) / (2 * np.pi ** 2 )) * (1 / (self.HBARC ** 3)) #needs to be divided by HBARC^3 to be in units of cm^-3

    def calculate_dist(self, state : State):
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, state.mn_eff, state.dm_eff = self.calculate_corrector_quantities(state)

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


        return [nue_n_dist.tolist(), anue_n_dist.tolist()], [nue_J_dist.tolist(), anue_J_dist.tolist()]

    #TODO: Something needs to be changed about this function and calculate_dist. We're evolving the neutrino spectrum, so the distribution should only be calculated at the first timestep.
    #As a result, temperature_from_e, get_potentials, and calculate_corrector_quantities need to be ran independently instead of inside calculate_dist.
    def calculate_spectral_rates(self, state : State):
        #Calculate number and energy density distributions
        state.nu_n_dist, state.nu_J_dist = self.calculate_dist(state)

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

        return rates

    def step(self, state: State, timestep : float):
        #Calculate number density distribution of incoming neutrinos
        nue_ext_n = self.nu_external_E[0] / self.E_dist #this may not technically be correct. This assumes that the prescribed distribution has the same energy bins as the calculated ones
        anue_ext_n = self.nu_external_E[1] / self.E_dist #same goes for here.

        E_before = np.sum(state.J_m1) + state.fluid_e + np.sum(self.nu_external_E[0]) + np.sum(self.nu_external_E[1]) #fluid energy density + total neutrino number density
        lep_num_before = (state.nb * state.ye) + np.sum(state.n_m1) + np.sum(nue_ext_n) + np.sum(anue_ext_n) #electron density + total neutrino density

        #calculate kappa for each species per energy bin, converting to inverse seconds
        kappa_nue = np.array([kappa * c.to("cm/s") for kappa in state.rates['kappa'][0]])
        kappa_anue = np.array([kappa * c.to("cm/s") for kappa in state.rates['kappa'][1]])

        #Find difference in number density distributions over the timestep
        nue_n_dist_diff = ((np.array(state.nu_n_dist[0]) + ((kappa_nue * timestep) * nue_ext_n)) / (1 + (kappa_nue * timestep))) - state.nu_n_dist[0]
        anue_n_dist_diff = ((np.array(state.nu_n_dist[1]) + ((kappa_anue * timestep) * anue_ext_n)) / (1 + (kappa_anue * timestep))) - state.nue_n_dist[1]

        #Find difference in baryon energy density over the timestep
        e_fluid_diff = -np.sum((nue_n_dist_diff + anue_n_dist_diff) * self.E_dist)

        #Find difference in electron fraction over the timestep
        ye_diff = (1 / state.nb) * np.sum(anue_n_dist_diff - nue_n_dist_diff)

        #TODO: Calculate total energy density and lepton number density after the step, find error, feed it back into the fluid.
        #TODO: Generate a new state after making the corrections for energy and lepton number error.
        return       
