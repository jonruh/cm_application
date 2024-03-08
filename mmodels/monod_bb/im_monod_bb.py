import numpy as np
from cmodels.sc_cmodel import sc_cmodel
import pandas as pd

class im_monod_bb(object):
    '''Description goes here.'''
    def __init__(self,strain_id:str=None,strain_description:str=None):
        self.strain_id = strain_id
        self.strain_description = strain_description

        # describe the simulated species
        self.species = pd.DataFrame(columns=['species','description','unit'])
        self.species['species'] = ['X','S','O','A']
        self.species['description'] = ['biomass concentration','substrate concentration','dissolved oxygen concentration','acetate concentration']
        self.species['unit'] = ['[g/L]','[g/L]','[mmol/L]','[g/L]']

        # describe the simulated rates
        self.rates = pd.DataFrame(columns=['rate','description','unit'])
        self.rates['rate'] = ['qS','mu','O_star','OTR','qO']
        self.rates['description'] = ['specific substrate uptake rate','specific growth rate','oxygen saturation concentration','oxygen transfer rate','specific oxygen uptake rate']
        self.rates['unit'] = ['[g/g/h]','[1/h]','[mmol/L]','[mmol/L/h]','[mmol/g/h]']

    def define_strain_params(self,qS_max,Ysx,Ks,Yso,Ko):
        '''Description goes here.'''
        self.mmodel_description = 'Simple Monod substrate uptake kinetic bloack box model. Oxygen limitation halts growth.'
        self.qS_max = qS_max
        self.Ysx    = Ysx
        self.Ks     = Ks
        self.Yso    = Yso
        self.Ko     = Ko
    
    def im_bmodel_odes(self,t,y,kla,pabs,yO,Sf:float,mu_set:float,V_fixed:bool=False,X_fixed:bool=False,D_set:float=0,pulse_cycle_time:float=None,pulse_on_ratio:float=None,returns:str='dydt'):
        '''Returns 1 out of 2 possible outputs: 
        1: A list of ODEs for a black box kinetic model. Used as "func" argument in scipy.integrate.odeint.
        2: List of arrays containing kinetic rate values of the black box kinetic model.
        
        t: array of timesteps to calculate solution for [h]
        y: array of ODE solutions
        kla: mass transfer coefficient for oxygen transfer [1/h]
        pabs: absolute pressure [bar]
        yO: oxygen mol fraction in the gas phase [-]
        Sf: substrate concentration of the feed solution [g/L]
        mu_set: Must be 0 when simulating batch process. Setpoint for the growth rate in chemostat or fed-batch processes. [1/h]
        V_fixed: Boolean value, if volume is fixed. True (batch, chemostat, pulse); False (fed-batch). Default False
        X_fixed: Boolean value, if biomass is fixed. True (pulse); False (batch, chemostat, fed-batch). Default False
        D_set: Setpoint for the dilution rate. Used only in pulse simulation. Default 0. [1/h]
        pulse_cycle_time: Time of one pulse cycle [s]. Used for pulse simulations. Default None
        pulse_on_ratio: Ratio of the feed pump being switched on during the pulse cycle time [-]. Used for pulse simulations. Default None
        returns: Specification what to return. 'dydt' or 'rates'
        '''

        # variables to solve (species)
        X =  y[0]
        S =  y[1]
        O =  y[2]
        A =  y[3]

        # rates equations
        ### qS = qS_max*S/(S+Ks)*O/(O+Ko) # [gS/gX/h] Specific substrate upteake rate
        qS = self.qS_max*S/(S+self.Ks)*O/(O+self.Ko)
        ### mu mu = qS*Ysx # [1/h] Sepcific growth rate
        mu = qS*self.Ysx
        ### O_star = yO*pabs # [mmol/L] oxygen concentration at the liquid gas interface
        O_star = yO*pabs 
        ### OTR = kLa*(O_star-O)*3600 # [mmol/L/h] oxygen transfer rate
        OTR = kla*(O_star-O)
        ### qO = qS*Yso # [mmol/gX/h] specific oxygen uptake rate
        qO = qS*self.Yso

        ### metabolic rates
        rates = [
            ['qS', qS],
            ['mu', mu],
            ['O_star', O_star],
            ['OTR', OTR],
            ['qO', qO]
            ]

        # the following settings are only needed when solving ODEs
        if returns == 'dydt':
            ### Dilution rate
            # pulse snapshot, chemostat or batch mode
            if V_fixed:
                # pulse snapshot
                if X_fixed:
                    pulse_on_time = pulse_on_ratio*pulse_cycle_time/3600
                    if t < pulse_on_time:
                        D = D_set/pulse_on_ratio
                    else:
                        D = 0
                # Batch or chemostat mode
                else:
                    D = mu_set
            # Fed-batch mode
            else:
                D = mu_set/self.Ysx*X/Sf
        

            ### ODEs for the species. Order of differential equations defined above
            # dXdt
            if X_fixed:
                dXdt = 0
            else:
                dXdt = X*(mu-D)
            # dSdt = dSdt_in - dSdt_out -qS*X + feed
            dSdt  = D*(Sf-S) - qS*X
            # dOdt = - qO*X + OTR
            dOdt  = OTR - qO*X
            # dAdt = 0
            dAdt  = 0

            # ODEs
            dydt = [
                dXdt,
                dSdt,
                dOdt,
                dAdt
            ]
        
        else:
            pass
        

        if returns == 'dydt':
            return dydt
        elif returns == 'rates':
            return rates
        else:    
            raise ValueError("argument 'returns' must be equal to either 'dydt' or 'rates'")