import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

class cell_tracking(object):

    def __init__(self,VL,CompFlows,t_eval,n_parcels):
        self.VL = VL # Liquid volumes, vector of liquid volumes of the compartment model [m3]
        self.V_relative = VL/VL.sum() # vector of relative volumes of the compartments
        self.CompFlows = CompFlows # Matrix of liquid exchange flows between compartments converted from [m3/s] into [m3/h]
        self.t_eval = t_eval # evaluation time for a duration of 10 min, with timesteps every 0.1s. Unit is in [h]
        self.t_step = t_eval[1] # timestep in [h]
        self.n_cells = n_parcels # number of cell to be tracked
        self.n_compartments = len(VL) # number of compartments in the compartment model
        self.compartments = np.array(range(1,self.n_compartments+1)) # array of compartment IDs ranging from 1 to n_compartments

    def calculate_cell_tracks(self):
        # initialize cell locations
        self.init_loc = np.random.choice(self.compartments,size=self.n_cells,p=self.V_relative).reshape(1,self.n_cells) # array of location of the tracked cells. A volume weighted random initial compartment is assigned.
        
        # The propability to jump to another compartment
        self.res_time = self.VL/self.CompFlows.sum(axis=1) # vector of residence times in the compartments, unit is in [h]
        self.P_jump = 1-np.exp(-self.t_step/self.res_time) # Probability of jumping to another compartment, exponential distribution
        self.P_stay = 1-self.P_jump

        # The probability to jump from compartment i (row) to compartment j (column) (in case of jumping)
        self.P_dest_jump = self.CompFlows / self.CompFlows.sum(axis=1)[:, np.newaxis]
        # The total probability to transition OR stay
        ## Multiply probabilities in case of jumping with the probability of jumping
        self.P_dest = self.P_dest_jump*self.P_jump[:, np.newaxis]
        ## Add the probability to stay in the compartment to the matrix
        for i in range(0,len(self.compartments)):
            self.P_dest[i,i] = self.P_stay[i]
        # The matrix P_dest (short for destination) is specific to the compartment flows, compartment volumes and the timestep selected for the cell tracking

        # Based on pre-computed probabilities, calculate lifelines

        cell_tracks_list = [] # list of all lifelines

        for init_comp in tqdm(self.init_loc[-1]):

            single_cell_track_list = [init_comp] # list of a single lifeline

            for time_step in self.t_eval[1:]:

                new_comp = np.random.choice(
                    a = self.compartments,
                    p = self.P_dest[single_cell_track_list[-1] - 1]
                )
            
                single_cell_track_list.append(new_comp)

            cell_tracks_list.append(single_cell_track_list)
            
        self.cell_tracks_list = cell_tracks_list

        # Test if the relative compartment volumes are represented by the cell tracking counts
        comp, comp_counts = np.unique(np.array(cell_tracks_list),return_counts=True)
        relative_volume = self.VL/self.VL.sum()
        relative_comp_counts = comp_counts/comp_counts.sum()

        df_cell_tracking = pd.DataFrame(columns=['comp','fraction','method'])
        df_cell_tracking['comp'] = comp
        df_cell_tracking['fraction'] = relative_comp_counts
        df_cell_tracking['method'] = ['cell tracking']*len(comp)

        df_volume = pd.DataFrame(columns=['comp','fraction','method'])
        df_volume['comp'] = comp
        df_volume['fraction'] = relative_volume
        df_volume['method'] = ['relative compartment model']*len(comp)

        df = pd.concat([df_cell_tracking,df_volume])

        sns.barplot(data=df, x="comp", y="fraction", hue="method").set(title='QC: Are relative cell tracking counts representing compartment volumes?')
    
    def couple_cmodel_solution(self,cmodel_sol):

        cell_tracks_array = np.array(self.cell_tracks_list)
        cell_lifelines = np.empty(shape=(cell_tracks_array.shape + cmodel_sol.sol_df.shape[1:]))

        for cell, lifeline in enumerate(cell_tracks_array):
            for step, comp in enumerate(lifeline):

                cell_lifelines[cell,step] = cmodel_sol.sol_df.iloc[int(comp)-1].values
                
        self.cell_lifelines = cell_lifelines