import numpy as np
import pandas as pd

class cc_xu_bb(object):
    '''Description goes here.'''
    def __init__(self,strain_id:str=None,strain_description:str=None):
        self.strain_id = strain_id
        self.strain_description = strain_description

        # describe the simulated species
        self.species = pd.DataFrame(columns=['species','description','unit'])
        self.species['species'] = [
            'S',
            'O',
            'A'
            ]
        self.species['description'] = [
            'substrate concentration',
            'dissolved oxygen concentration',
            'acetate concentration'
            ]
        self.species['unit'] = [
            '[g/L]',
            '[mmol/L]',
            '[g/L]'
            ]

        # describe the simulated rates
        self.rates = pd.DataFrame(columns=['rate','description','unit'])
        self.rates['rate'] = [
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
    
    def cc_cmodel_odes(self,t,y,comp_mod,X:float,Fs_feed:float,returns:str='dydt'):
        '''Description goes here.'''

        FeedDist = comp_mod.FeedDist
        CompFlows = np.array(comp_mod.CompFlows)*3600*1000 # [L/h]
        CompVolumes = comp_mod.CompVolumes
        GH = comp_mod.GH
        kla = comp_mod.kla # kla in 1/h
        O_star = comp_mod.O_star # in g/kg. Conversion from g/kg to mmol/L by dividing 32 g/mol and multiplying 1050 kg/m3 (later)
        VL = CompVolumes.CompVol - CompVolumes.CompVol*GH.GH

        dydt = []
        rates = []

        # Constructing the ODE system consisting of rate equations and differential equations
        compartments = len(CompVolumes)

        for n in range(0,compartments):
                
            # variables to solve (species)
            vars()['S'+str(n)] =  y[0 + n*3]
            vars()['O'+str(n)] =  y[1 + n*3]
            vars()['A'+str(n)] =  y[2 + n*3]

            # rates equations
            ### [mmol/L/h] oxygen transfer rate
            vars()['OTR'+str(n)] = kla.iloc[n].kla*(O_star.iloc[n].O_star/32*1050-vars()['O'+str(n)])
            ### [g/g/h] specific substrate uptake rate
            vars()['qS'+str(n)] = self.qS_max/(1+vars()['A'+str(n)]/self.Ki_s)*(vars()['S'+str(n)])/(vars()['S'+str(n)]+self.Ks)
            ### actual maintenance coefficient
            vars()['qm'+str(n)] = min(self.qm_max,vars()['qS'+str(n)])
            ### theoretical specific substrate uptake rate of the anabolic oxidative metabolism
            vars()['qS_ox_an_th'+str(n)] = (vars()['qS'+str(n)]-vars()['qm'+str(n)])*self.Ysx_ox*self.Cs/self.Cx
            ### theoretical specific substrate uptake rate of the energy oxidative metabolism
            vars()['qS_ox_en_th'+str(n)] = vars()['qS'+str(n)]-vars()['qS_ox_an_th'+str(n)]
            ### [mmol/g/h] oxygen uptake capacity compartment
            # Applying a Hill function to switch the maximum oxygen uptake capacity
            # When DO is still high, the oxygen uptake is only capped by the cells
            # But at low DO, oxygen uptake becomes capped by the reactors oxygen transfer capacity, i.e. OUR = OTR
            # Hence, a Hill function is introduced that switches to qO_max above 1% DO_airsat and OTC = kla*(O_star-1%DO_airsat) below 1% DO_airsat
            hill_coefficient = 100
            O_crit = 0.02*0.2095
            vars()['OUC'+str(n)] = (1-1/(1+np.power((O_crit/vars()['O'+str(n)]),hill_coefficient)))*vars()['OTR'+str(n)]/X + (1/(1+np.power((O_crit/vars()['O'+str(n)]),hill_coefficient)))*self.qO_max/(1+vars()['A'+str(n)]/self.Ki_o) # [mmol/gX/h] oxygen uptake capacity compartment 1
            ### specific oxygen uptake rate for substrate oxidation in the oxidative energy metabolism
            vars()['qO_s'+str(n)] = min(vars()['qS_ox_en_th'+str(n)]*self.Yso,vars()['OUC'+str(n)])
            ### specific substrate uptake rate of the energy oxidative metabolism
            vars()['qS_ox_en'+str(n)] = vars()['qO_s'+str(n)]/self.Yso
            ### the maintenance can not be higher than qS_ox_en. In case qS_ox_en <= qm, all substrate is only allocated to maintenance. In that case qm must be updated and brought to the level of qS_ox_en
            vars()['qm'+str(n)] = min(self.qm_max,vars()['qS_ox_en'+str(n)])
            ### specific substrate uptake rate of the anabolic oxidative metabolism
            vars()['qS_ox_an'+str(n)] = self.Ysx_ox*self.Cs/self.Cx*(vars()['qS_ox_en'+str(n)] - vars()['qm'+str(n)])/(1-self.Ysx_ox*self.Cs/self.Cx)
            ### specific substrate uptake rate of the oxidative metabolism
            vars()['qS_ox'+str(n)] = vars()['qS_ox_an'+str(n)] + vars()['qS_ox_en'+str(n)]
            ### specific substrate uptake rate of the overflow metabolism
            vars()['qS_of'+str(n)] = vars()['qS'+str(n)]-vars()['qS_ox'+str(n)]
            ### specific substrate uptake rate of the anabolic overflow metabolism
            vars()['qS_of_an'+str(n)] = vars()['qS_of'+str(n)]*self.Ysx_of*self.Cs/self.Cx
            ### specific substrate uptake rate of the energy overflow metabolism
            vars()['qS_of_en'+str(n)] = vars()['qS_of'+str(n)]-vars()['qS_of_an'+str(n)]
            ### specific acetate formation rate
            vars()['qA_p'+str(n)] = vars()['qS_of_en'+str(n)]*self.Ysa
            ### theoretical specific acetate consumption rate
            vars()['qA_c_th'+str(n)] = self.qA_c_max*vars()['A'+str(n)]/(vars()['A'+str(n)]+self.Ka) 
            ### theoretical specific acetate consumption rate for the anabolism
            vars()['qA_c_an_th'+str(n)] = vars()['qA_c_th'+str(n)]*self.Yax*self.Ca/self.Cx
            ### specific acetate consumption rate for the energy metabolism
            vars()['qA_c_en'+str(n)] = min(vars()['qA_c_th'+str(n)]-vars()['qA_c_an_th'+str(n)],(vars()['OUC'+str(n)]-vars()['qO_s'+str(n)])/self.Yao)
            ### specific acetate consumption rate for the anabolism
            vars()['qA_c_an'+str(n)] = vars()['qA_c_en'+str(n)]*(self.Yax*self.Ca/self.Cx)/(1-self.Yax*self.Ca/self.Cx)
            ### specific acetate consumption rate
            vars()['qA_c'+str(n)] = vars()['qA_c_an'+str(n)]+ vars()['qA_c_en'+str(n)]
            ### specific oxygen consumption rate
            vars()['qO'+str(n)] = vars()['qO_s'+str(n)]+vars()['qA_c_en'+str(n)]*self.Yao 
            ### specific biomass formation rate
            vars()['mu'+str(n)] = (vars()['qS_ox'+str(n)]-self.qm_max)*self.Ysx_ox+vars()['qS_of'+str(n)]*self.Ysx_of+vars()['qA_c'+str(n)]*self.Yax

            # append rates
            rates = np.append(
                rates,[
                    vars()['OTR'+str(n)],
                    vars()['qS'+str(n)], 
                    vars()['qm'+str(n)], 
                    vars()['qS_ox_an_th'+str(n)], 
                    vars()['qS_ox_en_th'+str(n)], 
                    vars()['OUC'+str(n)], 
                    vars()['qO_s'+str(n)], 
                    vars()['qS_ox_en'+str(n)], 
                    vars()['qS_ox_an'+str(n)], 
                    vars()['qS_ox'+str(n)], 
                    vars()['qS_of'+str(n)], 
                    vars()['qS_of_an'+str(n)], 
                    vars()['qS_of_en'+str(n)], 
                    vars()['qA_p'+str(n)], 
                    vars()['qA_c_th'+str(n)], 
                    vars()['qA_c_an_th'+str(n)], 
                    vars()['qA_c_en'+str(n)], 
                    vars()['qA_c_an'+str(n)], 
                    vars()['qA_c'+str(n)], 
                    vars()['qO'+str(n)], 
                    vars()['mu'+str(n)]
                ]
                )

        # define arrays with the shape (1,n) for each species
        S  = np.array([])
        O  = np.array([])
        A  = np.array([])

        for n in range(0,compartments):
            S = np.append(S,vars()['S'+str(n)])
            O = np.append(O,vars()['O'+str(n)])
            A = np.append(A,vars()['A'+str(n)])

        for n in range(0,compartments):

            # Infolows
            ### As a consequence of exchange flows between compartments, each compartment experiences inflows of 
            ### species from neighboring compartments. To simplify the notation of the ODEs, inflows are separately defined below.

            vars()['dSdt_in'+str(n)] = np.dot(S,CompFlows)[n] / (VL[n]*1000) # gS/L/h
            vars()['dOdt_in'+str(n)] = np.dot(O,CompFlows)[n] / (VL[n]*1000) # mmolO/L/h
            vars()['dAdt_in'+str(n)] = np.dot(A,CompFlows)[n] / (VL[n]*1000) # gA/L/h

            # Outflows
            ### As a consequence of exchange flows between compartments, each compartment experiences outflows of 
            ### species to neighboring compartments. To simplify the notation of the ODEs, outflows are separately defined below.

            vars()['dSdt_out'+str(n)] = np.sum(CompFlows,axis=1)[n] * vars()['S'+str(n)] / (VL[n]*1000) # gS/L/h
            vars()['dOdt_out'+str(n)] = np.sum(CompFlows,axis=1)[n] * vars()['O'+str(n)] / (VL[n]*1000) # mmolO/L/h
            vars()['dAdt_out'+str(n)] = np.sum(CompFlows,axis=1)[n] * vars()['A'+str(n)] / (VL[n]*1000) # gA/L/h

            # ODEs for the species. Order of differential equations defined above
            ### dSdt = dSdt_in - dSdt_out -qS*X + feed
            vars()['dSdt'+str(n)]  = vars()['dSdt_in'+str(n)] - vars()['dSdt_out'+str(n)] - vars()['qS'+str(n)] * X + FeedDist.FeedDist[n] * Fs_feed / (VL[n]*1000)
            ### dOdt = dOdt_in - dOdt_out - qO*X + OTR
            vars()['dOdt'+str(n)]  = vars()['dOdt_in'+str(n)] - vars()['dOdt_out'+str(n)] - vars()['qO'+str(n)] * X + vars()['OTR'+str(n)]
            ### dAdt = dAdt_in - dAdt_out - qA_c*X + qA_p*X
            vars()['dAdt'+str(n)]  = vars()['dAdt_in'+str(n)] - vars()['dAdt_out'+str(n)] + (vars()['qA_p'+str(n)] - vars()['qA_c'+str(n)]) * X

            # append ODEs
            dydt = np.append(
                dydt,[
                    vars()['dSdt'+str(n)],
                    vars()['dOdt'+str(n)],
                    vars()['dAdt'+str(n)]
                ]
                )

        if returns == 'dydt':
            return dydt
        elif returns == 'rates':
            return rates
        else:    
            raise ValueError("argument 'returns' must be equal to either 'dydt' or 'rates'")