import numpy as np
from cmodels.sc_cmodel import sc_cmodel
import pandas as pd

class im_xu_bb(object):
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
        self.rates['rate'] = [
            'O_star',
            'OTR',
            'qS', 
            'qm', 
            'qS_ox_an_th', 
            'qS_ox_en_th', 
            'OUC', 
            'qO_s', 
            'qS_ox_en', 
            'qS_ox_an', 
            'qS_ox', 
            'qS_of', 
            'qS_of_an', 
            'qS_of_en', 
            'qA_p', 
            'qA_c_th', 
            'qA_c_an_th', 
            'qA_c_en', 
            'qA_c_an', 
            'qA_c', 
            'qO', 
            'mu'
            ]

        self.rates['description'] = [
            'oxygen saturation concentration',
            'oxygen transfer rate',
            'specific substrate uptake rate',
            'actual maintenance coefficient',
            'theoretical specific substrate uptake rate of the anabolic oxidative metabolism',
            'theoretical specific substrate uptake rate of the energy oxidative metabolism',
            'oxygen uptake capacity',
            'specific oxygen uptake rate for substrate oxidation in the oxidative energy metabolism',
            'specific substrate uptake rate of the energy oxidative metabolism',
            'specific substrate uptake rate of the anabolic oxidative metabolism',
            'specific substrate uptake rate of the oxidative metabolism',
            'specific substrate uptake rate of the overflow metabolism',
            'specific substrate uptake rate of the anabolic overflow metabolism',
            'specific substrate uptake rate of the energy overflow metabolism',
            'specific acetate formation rate',
            'theoretical specific acetate consumption rate',
            'theoretical specific acetate consumption rate for the anabolism',
            'specific acetate consumption rate for the energy metabolism',
            'specific acetate consumption rate for the anabolism',
            'specific acetate consumption rate',
            'specific oxygen consumption rate',
            'specific biomass formation rate'
            ]

        self.rates['unit'] = [
            '[mmol/L]',
            '[mmol/L/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[mmol/g/h]',
            '[mmol/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[g/g/h]',
            '[mmol/g/h]',
            '[1/h]'
        ]

    def define_strain_params(self,qS_max,qm_max,qA_c_max,qO_max,Ysx_ox,Ysx_of,Ysa,Yax,Ki_s,Ks,Ka,Ki_o):
        '''Description goes here.'''
        self.mmodel_description = 'Black box model of E. coli overflow metabolism adapted from Xu et al. 1999. Substrate uptake modelled as monod kinetic.'
        
        # stoichiometric constants
        self.Cs = 30    # [gS/CmolS]
        self.Cx = 24.6  # [gX/CmolX]
        self.Ca = 30    # [gA/CmolA]
        self.Yso = 33.3 # [mmolO/gS]
        self.Yao = 33.3 # [mmolO/gS]

        # physiological parameters
        self.qS_max      = qS_max  # [gS/gX/h]    
        self.qm_max      = qm_max  # [gS/gX/h] maximum maintenance coefficient. If substrate is severely limited, qm must be lower than this value, to not cause mass conservation issues in the model
        self.qA_c_max    = qA_c_max# [gA/gX/h]
        self.qO_max      = qO_max  # [mmolO/gX/h]
        
        self.Ysx_ox      = Ysx_ox # [gX/gS]
        self.Ysx_of      = Ysx_of # [gX/gS]
        self.Ysa         = Ysa    # [gA/gS]
        self.Yax         = Yax    # [gX/gA]
        
        self.Ki_s        = Ki_s   # [gA/L]
        self.Ks          = Ks     # [gS/L]
        self.Ka          = Ka     # [gA/L]
        self.Ki_o        = Ki_o   # [gA/L]
    
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
        ### O_star = yO*pabs # [mmol/L] oxygen concentration at the liquid gas interface
        O_star = yO*pabs 
        ### [mmol/L/h] oxygen transfer rate
        OTR = kla*(O_star-O)
        ### [g/g/h] specific substrate uptake rate
        qS = self.qS_max/(1+A/self.Ki_s)*(S)/(S+self.Ks)
        ### actual maintenance coefficient
        qm = np.minimum(self.qm_max,qS)
        ### theoretical specific substrate uptake rate of the anabolic oxidative metabolism
        qS_ox_an_th = (qS-qm)*self.Ysx_ox*self.Cs/self.Cx
        ### theoretical specific substrate uptake rate of the energy oxidative metabolism
        qS_ox_en_th = qS-qS_ox_an_th
        ### [mmol/g/h] oxygen uptake capacity compartment
        # Applying a Hill function to switch the maximum oxygen uptake capacity
        # When DO is still high, the oxygen uptake is only capped by the cells
        # But at low DO, oxygen uptake becomes capped by the reactors oxygen transfer capacity, i.e. OUR = OTR
        # Hence, a Hill function is introduced that switches to qO_max above 1% DO_airsat and OTC = kla*(O_star-1%DO_airsat) below 1% DO_airsat
        hill_coefficient = 100
        O_crit = 0.02*0.2095 # 2% of air saturation
        OUC = (1-1/(1+np.power((O_crit/O),hill_coefficient)))*OTR/X + (1/(1+np.power((O_crit/O),hill_coefficient)))*self.qO_max/(1+A/self.Ki_o) # [mmol/gX/h] oxygen uptake capacity compartment 1
        ### specific oxygen uptake rate for substrate oxidation in the oxidative energy metabolism
        qO_s = np.minimum(qS_ox_en_th*self.Yso,OUC)
        ### specific substrate uptake rate of the energy oxidative metabolism
        qS_ox_en = qO_s/self.Yso
        ### the maintenance can not be higher than qS_ox_en. In case qS_ox_en <= qm, all substrate is only allocated to maintenance. In that case qm must be updated and brought to the level of qS_ox_en
        qm = np.minimum(self.qm_max,qS_ox_en)
        ### specific substrate uptake rate of the anabolic oxidative metabolism
        qS_ox_an = self.Ysx_ox*self.Cs/self.Cx*(qS_ox_en - qm)/(1-self.Ysx_ox*self.Cs/self.Cx)
        ### specific substrate uptake rate of the oxidative metabolism
        qS_ox = qS_ox_an + qS_ox_en
        ### specific substrate uptake rate of the overflow metabolism
        qS_of = qS - qS_ox
        ### specific substrate uptake rate of the anabolic overflow metabolism
        qS_of_an = qS_of*self.Ysx_of*self.Cs/self.Cx
        ### specific substrate uptake rate of the energy overflow metabolism
        qS_of_en = qS_of - qS_of_an
        ### specific acetate formation rate
        qA_p = qS_of_en*self.Ysa
        ### theoretical specific acetate consumption rate
        qA_c_th = self.qA_c_max*A/(A+self.Ka) 
        ### theoretical specific acetate consumption rate for the anabolism
        qA_c_an_th = qA_c_th*self.Yax*self.Ca/self.Cx
        ### specific acetate consumption rate for the energy metabolism
        qA_c_en = np.minimum(qA_c_th - qA_c_an_th,(OUC - qO_s)/self.Yao)
        ### specific acetate consumption rate for the anabolism
        qA_c_an = qA_c_en * (self.Yax*self.Ca/self.Cx)/(1-self.Yax*self.Ca/self.Cx)
        ### specific acetate consumption rate
        qA_c = qA_c_an + qA_c_en
        ### specific oxygen consumption rate
        qO = qO_s + qA_c_en*self.Yao 
        ### specific biomass formation rate
        mu = (qS_ox-self.qm_max)*self.Ysx_ox + qS_of * self.Ysx_of + qA_c * self.Yax

        # append rates
        rates = [
            ['O_star',O_star],
            ['OTR',OTR],
            ['qS',qS], 
            ['qm',qm], 
            ['qS_ox_an_th',qS_ox_an_th], 
            ['qS_ox_en_th',qS_ox_en_th], 
            ['OUC',OUC], 
            ['qO_s',qO_s], 
            ['qS_ox_en',qS_ox_en], 
            ['qS_ox_an',qS_ox_an], 
            ['qS_ox',qS_ox], 
            ['qS_of',qS_of], 
            ['qS_of_an',qS_of_an], 
            ['qS_of_en',qS_of_en], 
            ['qA_p',qA_p], 
            ['qA_c_th',qA_c_th], 
            ['qA_c_an_th',qA_c_an_th], 
            ['qA_c_en',qA_c_en], 
            ['qA_c_an',qA_c_an], 
            ['qA_c',qA_c], 
            ['qO',qO], 
            ['mu',mu]
            ]

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
                D = mu_set/self.Ysx_ox*X/Sf
        

            ### ODEs for the species. Order of differential equations defined above
            # dXdt
            if X_fixed:
                dXdt = 0
            else:
                dXdt = X*(mu-D)
            # dSdt
            dSdt  = D*(Sf-S) - qS*X
            # dOdt
            dOdt  = OTR - qO*X
            # dAdt
            dAdt  = (qA_p-qA_c)*X - D*A

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