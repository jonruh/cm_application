import numpy as np
from cmodels.sc_cmodel import sc_cmodel
import pandas as pd

class sc_monod_bb(object):
    '''Description goes here.'''
    def __init__(self,strain_id:str=None,strain_description:str=None):
        self.strain_id = strain_id
        self.strain_description = strain_description

        # describe the simulated species
        self.species = pd.DataFrame(columns=['species','description','unit'])
        self.species['species'] = ['S','O','A','yO']
        self.species['description'] = ['substrate concentration','dissolved oxygen concentration','acetate concentration','oxygen molar fraction']
        self.species['unit'] = ['[g/L]','[mmol/L]','[g/L]','[-]']

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
    
    def sc_cmodel_odes(self,t,y,cmodel:sc_cmodel,X:float,Fs_feed:float,returns:str='dydt'):
        '''Description goes here.'''
        comp_flows, VL, kLa, pabs, feed_dist, ng = cmodel.calc_sc_cmodel()
        Fg    = cmodel.Fg
        MW_g  = cmodel.MW_g
        yO_in = cmodel.yO_in
        n_comps = cmodel.n_comps

        dydt = []
        rates = []

        # Constructing the ODE system consisting of rate equations and differential equations
        compartments = n_comps

        for n in range(0,compartments):
                
            # variables to solve (species)
            vars()['S'+str(n)] =  y[0 + n*4]
            vars()['O'+str(n)] =  y[1 + n*4]
            vars()['A'+str(n)] =  y[2 + n*4]
            vars()['yO'+str(n)] = y[3 + n*4]

            # rates equations
            ### qS = qS_max*S/(S+Ks)*O/(O+Ko) # [gS/gX/h] Specific substrate upteake rate
            vars()['qS'+str(n)] = self.qS_max*vars()['S'+str(n)]/(vars()['S'+str(n)]+self.Ks)*vars()['O'+str(n)]/(vars()['O'+str(n)]+self.Ko)
            ### mu mu = qS*Ysx # [1/h] Sepcific growth rate
            vars()['mu'+str(n)] = vars()['qS'+str(n)]*self.Ysx
            ### O_star = yO*pabs # [mmol/L] oxygen concentration at the liquid gas interface
            vars()['O_star'+str(n)] = vars()['yO'+str(n)]*pabs[n] 
            ### OTR = kLa*(O_star-O)*3600 # [mmol/L/h] oxygen transfer rate
            vars()['OTR'+str(n)] = kLa[n]*(vars()['O_star'+str(n)]-vars()['O'+str(n)])*3600 
            ### qO = qS*Yso # [mmol/gX/h] specific oxygen uptake rate
            vars()['qO'+str(n)] = vars()['qS'+str(n)]*self.Yso

            # append rates
            rates = np.append(
                rates,[
                    vars()['qS'+str(n)],
                    vars()['mu'+str(n)],
                    vars()['O_star'+str(n)],
                    vars()['OTR'+str(n)],
                    vars()['qO'+str(n)]
                ]
                )

        # define arrays with the shape (1,n) for each species
        S  = np.array([])
        O  = np.array([])
        A  = np.array([])
        yO = np.array([])

        for n in range(0,compartments):
            S = np.append(S,vars()['S'+str(n)])
            O = np.append(O,vars()['O'+str(n)])
            A = np.append(A,vars()['A'+str(n)])
            yO = np.append(yO,vars()['yO'+str(n)])
        yO = np.append(yO,yO_in)  # special case for yO to account for yO "below" the bottom compartment

        for n in range(0,compartments):

            # Infolows
            ### As a consequence of exchange flows between compartments, each compartment experiences inflows of 
            ### species from neighboring compartments. To simplify the notation of the ODEs, inflows are separately defined below.

            vars()['dSdt_in'+str(n)] = np.dot(S,comp_flows)[n] / (VL[n]*1000) # gS/L/h
            vars()['dOdt_in'+str(n)] = np.dot(O,comp_flows)[n] / (VL[n]*1000) # mmolO/L/h
            vars()['dAdt_in'+str(n)] = np.dot(A,comp_flows)[n] / (VL[n]*1000) # gA/L/h

            # Outflows
            ### As a consequence of exchange flows between compartments, each compartment experiences outflows of 
            ### species to neighboring compartments. To simplify the notation of the ODEs, outflows are separately defined below.

            vars()['dSdt_out'+str(n)] = np.sum(comp_flows,axis=1)[n] * vars()['S'+str(n)] / (VL[n]*1000) # gS/L/h
            vars()['dOdt_out'+str(n)] = np.sum(comp_flows,axis=1)[n] * vars()['O'+str(n)] / (VL[n]*1000) # mmolO/L/h
            vars()['dAdt_out'+str(n)] = np.sum(comp_flows,axis=1)[n] * vars()['A'+str(n)] / (VL[n]*1000) # gA/L/h

            # ODEs for the species. Order of differential equations defined above
            ### dSdt = dSdt_in - dSdt_out -qS*X + feed
            vars()['dSdt'+str(n)]  = vars()['dSdt_in'+str(n)] - vars()['dSdt_out'+str(n)] - vars()['qS'+str(n)] * X + feed_dist[n] * Fs_feed / (VL[n]*1000)
            ### dOdt = dOdt_in - dOdt_out - qO*X + OTR
            vars()['dOdt'+str(n)]  = vars()['dOdt_in'+str(n)] - vars()['dOdt_out'+str(n)] - vars()['qO'+str(n)] * X + vars()['OTR'+str(n)]
            ### dAdt = 0
            vars()['dAdt'+str(n)]  = 0
            ### dyOdt = (molO2_in - molO2_out)/mol_total (Assuming RQ = 1) 
            vars()['dyOdt'+str(n)] = (Fg/MW_g * 3600 * (yO[n+1] - yO[n]) - vars()['OTR'+str(n)] * VL[n]) / ng[n]

            # append ODEs
            dydt = np.append(
                dydt,[
                    vars()['dSdt'+str(n)],
                    vars()['dOdt'+str(n)],
                    vars()['dAdt'+str(n)],
                    vars()['dyOdt'+str(n)]
                ]
                )

        if returns == 'dydt':
            return dydt
        elif returns == 'rates':
            return rates
        else:    
            raise ValueError("argument 'returns' must be equal to either 'dydt' or 'rates'")

    def cstarv_ifr_sd_odes(self,t,y,X:float,Sf:float,mu_set:float,pulse_cycle_time:float,pulse_on_ratio:float):
        S = y

        D_set = mu_set/self.Ysx*X/Sf
        pulse_on_time = pulse_on_ratio*pulse_cycle_time/3600

        if t < pulse_on_time:
            D = D_set/pulse_on_ratio
        else:
            D = 0

        qS = self.qS_max*S/(S+self.Ks)

        dydt = -qS*X+D*(Sf-S)

        return dydt