class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, nu_n_dist = None, chi_m1 = [1/3, 1/3, 1/3, 1/3], time = 0.0, dU = None):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV cm-3]
        self.n_m1 = None
        self.J_m1 = None
        self.nu_n_dist = nu_n_dist
        self.nu_J_dist = None                                  
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.ndot_sources = None                                 # Neutrino number density source terms for integration [cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None                                  # Dirac effective proton mass [MeV]
        self.mn_eff = None                                  # Dirac effective neutron mass [MeV]
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = dU
        self.time = time                                    # Time at which the state occurs during the simulation [s]
