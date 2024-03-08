import numpy as np
import pandas as pd

class cc_monod_bb(object):
    '''Description goes here.'''
    def __init__(self,strain_id:str=None,strain_description:str=None):
        self.strain_id = strain_id
        self.strain_description = strain_description

        # describe the simulated species
        self.species = pd.DataFrame(columns=['species','description','unit'])
        self.species['species'] = ['S','O','A']
        self.species['description'] = ['substrate concentration','dissolved oxygen concentration','acetate concentration']
        self.species['unit'] = ['[g/L]','[mmol/L]','[g/L]']

        # describe the simulated rates
        self.rates = pd.DataFrame(columns=['rate','description','unit'])
        self.rates['rate'] = ['qS','mu','OTR','qO']
        self.rates['description'] = ['specific substrate uptake rate','specific growth rate','oxygen transfer rate','specific oxygen uptake rate']
        self.rates['unit'] = ['[g/g/h]','[1/h]','[mmol/L/h]','[mmol/g/h]']

    def define_strain_params(self,qS_max,Ysx,Ks,Yso,Ko):
        '''Description goes here.'''
        self.mmodel_description = 'Simple Monod substrate uptake kinetic bloack box model. Oxygen limitation halts growth.'
        self.qS_max = qS_max
        self.Ysx    = Ysx
        self.Ks     = Ks
        self.Yso    = Yso
        self.Ko     = Ko
    
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
            ### qS = qS_max*S/(S+Ks)*O/(O+Ko) # [gS/gX/h] Specific substrate upteake rate
            vars()['qS'+str(n)] = self.qS_max*vars()['S'+str(n)]/(vars()['S'+str(n)]+self.Ks)*vars()['O'+str(n)]/(vars()['O'+str(n)]+self.Ko)
            ### mu mu = qS*Ysx # [1/h] Sepcific growth rate
            vars()['mu'+str(n)] = vars()['qS'+str(n)]*self.Ysx
            ### OTR = kLa*(O_star-O) # [mmol/L/h] oxygen transfer rate
            vars()['OTR'+str(n)] = kla.iloc[n].kla*(O_star.iloc[n].O_star/32*1050-vars()['O'+str(n)])
            ### qO = qS*Yso # [mmol/gX/h] specific oxygen uptake rate
            vars()['qO'+str(n)] = vars()['qS'+str(n)]*self.Yso

            # append rates
            rates = np.append(
                rates,[
                    vars()['qS'+str(n)],
                    vars()['mu'+str(n)],
                    vars()['OTR'+str(n)],
                    vars()['qO'+str(n)]
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
            ### dAdt = 0
            vars()['dAdt'+str(n)]  = 0

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