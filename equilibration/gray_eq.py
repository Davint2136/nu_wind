import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy


class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1, dU):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV cm-3]
        self.n_m1 = n_m1                                    # Gray neutrino number densities. Order: nue, anue, nux, anux [cm-3]
        self.J_m1 = J_m1                                    # Gray neutrino energy densities. Order: nue, anue, nux, anux [MeV cm-3]
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.source_terms = {"edot" : None, "ndot" : None}  # Neutrino transport source terms [MeV cm-3 s-1, cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None
        self.mn_eff = None
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = dU                                      # Nucleon interaction potential difference [MeV]


class Solver:
    def __init__(self, table, state, integrate_tolerance, opacity_flags, opacity_pars):
        self.timestep = None
        self.integrate_tolerance = integrate_tolerance
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars


    def temperature_from_e(self, nb : float, ye : float, e_val : float):
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
                nb [float]: Baryon number density in cm^-3.
                ye [float]: Unitless charge fraction.
                e_val [float]: The fluid's energy density in MeV/cm^3.

            Outputs:
                t [float]: The fluid temperature in MeV.
        """
        from scipy.optimize import bisect

        # Find difference between calculated e and actual e. Ensure lengths are in fm for PyCompOSE compatibility.
        def f(t):
            interp = self.table.interpolate(np.array([nb]), np.array([ye]), np.array([t]), method='linear')
            e_table = (interp.thermo["Q7"] + 1) * interp.mn * nb
            return e_val - e_table[0, 0, 0]


        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")
        
        return t
    
    def get_potentials(self):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """
        current_state = self.states[-1]

        interp = self.table.interpolate_3D(np.array([current_state.nb]) * 1e-39, np.array([current_state.ye]), np.array([current_state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self):
        # TODO: Use eos.micro to find dirac effective masses to calculate dm_eff, don't attempt dU for now
        current_state = self.states[-1]

        interp = self.table.interpolate_3D(np.array([current_state.nb]) * 1e-39, np.array([current_state.ye]), np.array([current_state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff
        return mp_eff, mn_eff, dm_eff

    def calculate_gray_rates(self):
        # TODO: Extremely similar to bns_nurates' test_bindings.py. Use the same code structure
        
        #Initialize eos_pars and populate
        current_state = self.states[-1]
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = current_state.nb * 1e-21 # Convert baryon number density to nm^-3
        eos_pars.temp = current_state.t
        eos_pars.ye = current_state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = current_state.mu_n
        eos_pars.mu_e = current_state.mu_e
        eos_pars.mu_p = current_state.mu_p
        eos_pars.dm_eff = current_state.dm_eff
        eos_pars.dU = current_state.dU

        #Create a quadrature, populate it with data for a Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 6
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars

        #Load M1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = current_state.chi_m1
        m1_pars.n = [x * 1e-21 for x in current_state.n_m1]
        m1_pars.J = [x * 1e-21 for x in current_state.J_m1]

        distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)

        #Populate global structure using grey_pars
        grey_pars = bns.GreyOpacityParams()
        grey_pars.eos_pars = eos_pars
        grey_pars.opacity_flags = opacity_flags
        grey_pars.opacity_pars = opacity_pars
        grey_pars.distr_pars = distr_pars
        grey_pars.m1_pars = m1_pars

        #Calculate rates
        gray_rates = bns.ComputeM1Opacities(quad, quad, grey_pars)
        gray_rates['eta']       = [x * 1e21 for x in gray_rates['eta']]
        gray_rates['eta_0']     = [x * 1e21 for x in gray_rates['eta_0']]
        gray_rates['kappa_a']   = [x * 1e7  for x in gray_rates['kappa_a']]
        gray_rates['kappa_0_a'] = [x * 1e7  for x in gray_rates['kappa_0_a']]
        gray_rates['kappa_s']   = [x * 1e7  for x in gray_rates['kappa_s']]
        return gray_rates

    def calculate_source_terms(self):
        rates = self.states[-1].rates
        J_m1 = self.states[-1].J_m1
        n_m1 = self.states[-1].n_m1

        e_terms = [None, None, None, None]
        n_terms = [None, None, None, None]

        for i in range(0, 4):
            edot = rates["eta"][i] - (rates["kappa_a"][i] * J_m1[i])
            ndot = rates["eta_0"][i] - (rates["kappa_0_a"][i] * n_m1[i])
        

        return {"edot" : edot, "ndot" : ndot}

    def integrate_step(self):
        # TODO: Build and test RK2 integrator before implementing anything here. Best to do this in a separate file.
        pass

    #TODO: Add a create_state function to make State objects?



#TESTING-------------------------------------------------------------
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))

md = Metadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads = {
        999: ("N", "average nucleous")
    },
    micro={
        10041: ("mn_d", "neutron dirac effective mass divided by neutron mass"),
        11041: ("mp_d", "proton dirac effective mass divided by proton mass")
    }
)
eos = Table(md)
print("reading...")
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

## Input thermodynamic quantities (corresponding to point A in Chiesa+25 PRD)
##  N.B.: chemical potentials include the rest mass contribution
nb   = 4.208366627847035e+38  # Baryon number density [cm-3]      
e_nb = 4.3441769719367175e+41 # Baryon energy density [MeV cm-3]
temp = 12.406403541564941     # Temperature [MeV]
ye   = 0.07158458232879639    # Electron fraction
xn   = 1. - ye                # Neutron fraction
xp   = ye                     # Proton fraction
mu_e = 187.1814489            # Electron chemical potential [MeV]
mu_p = 1011.01797737          # Proton chemical potential [MeV]
mu_n = 1221.59013681          # Neutron chemical potential [MeV]
dU = 18.92714728              # Nucleon interaction potential difference (Un-Up) [MeV]
mp_eff = 278.87162217         # Proton effective mass [MeV]
mn_eff = 280.16495513         # Neutron effective mass [MeV]
dm = mn_eff - mp_eff          # Nucleon effective mass difference [MeV]

n_m1 = [3.739749408027436e+33, 1.2174961961689319e+35, 2.2438496448164613e+34, 2.2438496448164613e+34]  # Neutrino number densities [cm-3]
J_m1 = [1.246583136009145e+35,  5.360307484839323e+36,  8.726081952064015e+35,  8.726081952064015e+35]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1./3., 1./3., 1./3., 1./3.]  # Eddington factor

init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': True, 'use_dm_eff': True, 'use_WM_ab': True, 'use_WM_sc': True, 'use_decay': True, 'brem_implementation': 'HR98', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state, None, opacity_flags, opacity_pars)
solver.states[-1].t = solver.temperature_from_e(solver.states[-1].nb * 1e-39, solver.states[-1].ye, solver.states[-1].fluid_e * 1e-39)

"""TODO: temperature_from_var is close, but not close enough for my liking. For point A of Chiesa et al. 2025, the difference between the computed
         temperature and the actual temperature is around 0.034 MeV. This is close, but provided that some quantities like neutrino energy and number
         densities are sensitive to temperature, I'd like the difference to be smaller. Changing scipy.optimize.bisect's tolerances should help, 
         but the performance impact needs to be measured as well.

    NOTE: Changing the tolerances did nothing. This is most likely an interpolation error issue. I'll leave this be for now.
"""

solver.states[-1].mu_p, solver.states[-1].mu_n, solver.states[-1].mu_e = solver.get_potentials()
solver.states[-1].mp_eff, solver.states[-1].mn_eff, solver.states[-1].dm_eff = solver.calculate_corrector_quantities()
print(solver.states[-1].mp_eff, solver.states[-1].mn_eff, solver.states[-1].dm_eff) #These differ from Point A of Chiesa et al. 2025, dm_eff should be orders of magnitude larger. EOS related? Interpolator error?
solver.states[-1].rates = solver.calculate_gray_rates()
bns.print_integrated_rates(solver.states[-1].rates)

