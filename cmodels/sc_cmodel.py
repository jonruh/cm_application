import math
import numpy as np
import pandas as pd

class sc_cmodel(object):
    '''Description goes here.'''

    p_atm = 1.013 # [bar] Atmospheric pressure
    R = 8.3145 # [J mol-1 K-1] Ideal gas constant
    MW_g = 0.02897 # [kg mol-1] Molecular weight of air
    dyn_visc = 0.001 # [Pa s] Dynamic viscosity of water
    dens_L = 1050 # [kg m-3] Broth density
    interf_tension = 0.071 # [N/m] Interfacial tension of water at 30 degrees celsius
    D_L = 2.42E-09 # [m2 s-1] Diffusivity of oxygen on the liquid
    
    def __init__(
        self,
        # Exel file of the compartment map
        excel_path,
        # process variables
        N_rpm, Fg, p_head, Temp_C:float=30.0, db_out:float=0.005, yO_in:float=0.2095, 
        # compartment settings
        feed_comps:list=[1], feed_comp_props:list=[1],
        # tank gemotery
        T:float=3.476, H:float=10.464, n:int=4, D:float=0.936, L:float=2.32, L1:float=1.271, Np:float=6.0
        ):

        self.N = N_rpm/60
        self.Fg = Fg
        self.p_head = p_head
        self.Temp = Temp_C+273.15
        self.db_out = db_out
        self.yO_in = yO_in
        
        self.comp_vols = np.array([30,20,20,20])
        self.n_comps = 4
        self.comps = np.array([1,2,3,4])
        self.feed_comps = np.array(feed_comps)
        self.feed_comp_props = feed_comp_props

        self.T = T
        self.A = np.power(T/2,2) * math.pi
        self.H = H 
        self.n = n
        self.D = D
        self.L = L
        self.L1 = L1
        self.Np = Np

        self.Fr = np.power(self.N,2)*self.D/9.81 # [-] Froude number
        self.Re = self.dens_L*self.N*np.power(self.D,2)/self.dyn_visc

        self.CompMap = pd.read_excel(excel_path,sheet_name='CompMap')
        
    def calc_sc_cmodel(self):
        '''Description goes here.'''

        self.pabs   = (self.dens_L*9.81*(self.H-(4-self.comps)*self.L-self.L1))/100000+self.p_head+self.p_atm # [bar] Absolute pressure, height of the impeller
        self.dens_g = self.pabs*100000*self.MW_g/self.R/self.Temp # [kg m-3] Gas density
        self.Q_g = self.Fg/self.dens_g # [m3 s-1] Volumetric gas flow
        self.vg_s = self.Q_g/self.A # [m s-1] Superficial gas velocity
        # hold up calculation according to Garcia Ochoa, not used 
        # self.Vg_by_VL = 0.819*np.power(self.vg_s,0.66)*np.power(self.N,0.4)*np.power(self.D,(4/15))/np.power(9.81,0.33)*np.power((self.dens_L/self.interf_tension),0.2)*(self.dens_L/(self.dens_L-self.dens_g))*np.power((self.dens_L/self.dens_g),(-1/15)) # [-] ε/(1-ε) Gas hold-up relative to liquid volume
        # self.hold_up = self.Vg_by_VL/(self.Vg_by_VL+1) # [-] Gas hold-up
        
        self.Pg_by_P0 = 0.0312*np.power(self.Fr,-0.16)*np.power(self.Re,0.064)*np.power((self.Q_g/self.N/np.power(self.D,3)),-0.38)*np.power(self.T/self.D,0.8) # [-] Ratio between aerated and unaerated power input by stirring @ average gas flow and pressure
        self.Ps = self.Np*self.Pg_by_P0*self.dens_L*np.power(self.N,3)*np.power(self.D,5) # [W] Power input from stirring per impeller
        
        # hold up according to Noorman et al.
        self.hold_up = 0.13 * np.power(self.Ps/self.comp_vols,1/3) * np.power(self.vg_s,2/3)
        
        self.VL = self.comp_vols*(1-self.hold_up) # [m3] Liquid volume
        self.ng = (self.comp_vols-self.VL)*self.dens_g/self.MW_g # [mol] Mols of gas present in compartment
        self.FP_g = 0.2*self.Np*self.Pg_by_P0*self.N*np.power(self.D,3) # [m3 s-1] Pumping capacity, aerated
        # kla according to Garcia-Ochoa (not in use)
        # self.kL = 2/np.power(math.pi,0.5)*np.power(self.D_L,0.5)*np.power((self.Ps/(math.pi/4*np.power(self.D,2)*(self.comp_vols/self.A))),0.25) # [m s-1]
        # self.db = self.db_out*np.power(((self.p_atm+self.p_head)/self.pabs),(1/3)) # [m] bubble diameter
        # self.a = 6*self.hold_up/self.db # [m-1] 
        # self.kLa = self.kL * self.a # [s-1] Mass transfer coefficient kLa, coalescing
        
        # kla of non-coalescing bubbles according to Noorman et al.
        self.kLa = 0.002 * np.power(self.Ps/self.VL, 0.7) * np.power(self.vg_s, 0.2)
        
        # Dividing by 4, becuase half of pumping flow is diverted into the direction of the neighboring compartment and the average between 2 flows is considered
        Fex_g_12 = (self.FP_g[0] + self.FP_g[1])/4 *3600*1000 # [L h-1] Exchange flow between compartment 1 and 2
        Fex_g_23 = (self.FP_g[1] + self.FP_g[2])/4 *3600*1000 # [L h-1] Exchange flow between compartment 2 and 3
        Fex_g_34 = (self.FP_g[2] + self.FP_g[3])/4 *3600*1000 # [L h-1] Exchange flow between compartment 3 and 4

        # building a matrix (m,n) of exchnage flows, where at position (m,n) 
        # the flow from m to n is denoted 
        self.comp_flows = np.zeros((self.n_comps,self.n_comps))
        
        self.comp_flows[0,1], self.comp_flows[1,0] = [Fex_g_12]*2
        self.comp_flows[1,2], self.comp_flows[2,1] = [Fex_g_23]*2
        self.comp_flows[2,3], self.comp_flows[3,2] = [Fex_g_34]*2

        self.feed_dist = np.zeros(self.n_comps)
        for i in range(0,len(self.feed_comps)):
            feed_comp = self.feed_comps[i]
            feed_comp_prop = self.feed_comp_props[i]
            self.feed_dist[feed_comp-1] = feed_comp_prop

        return [self.comp_flows,self.VL,self.kLa,self.pabs,self.feed_dist,self.ng]