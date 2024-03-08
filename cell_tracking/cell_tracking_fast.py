import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

class cell_tracking_fast(object):

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
        self.P_jump = 1-np.exp(-self.t_step/self.res_time)

        # The probability to jump from compartment i (row) to compartment j (column) (in case of jumping)
        P_dest_jump = self.CompFlows / self.CompFlows.sum(axis=1)[:, np.newaxis]

        # Add 1.0 diagonally to P_dest_jump to facilitate later computation
        diagonal_ones = np.diag(np.diag(np.ones((self.n_compartments,self.n_compartments))))
        P_dest = P_dest_jump+diagonal_ones
        
        # these are the sorting indices
        sorted_idx = np.flip(np.argsort(P_dest),axis=1)

        # these are the cumsums from descending P_dest values
        sorted_P_dest_cumsum = np.cumsum(np.take_along_axis(arr=P_dest,indices=sorted_idx,axis=1),axis=1)
        sorted_P_dest_cumsum = sorted_P_dest_cumsum-1 # subtracting the 1 again

        # iterate over timesteps to compute cell tracks

        new_loc_temp = self.init_loc
        loc = new_loc_temp
        cell_tracks_list = []

        for step in tqdm(self.t_eval):

            # jump quantifier
            Q_jump = (self.P_jump[new_loc_temp[-1]-1]-np.random.rand(self.n_cells))/self.P_jump[new_loc_temp[-1]-1]
            # a is the zero-indexed ID corresponding to the sorted cumsums
            a = np.diag(np.apply_along_axis(np.searchsorted, 1, sorted_P_dest_cumsum[new_loc_temp[-1]-1], Q_jump))
            # new_loc are the new compartment locations (+1, because compartment IDs are not zero-indexed)
            new_loc = sorted_idx[new_loc_temp[-1]-1,a] + 1
            new_loc = new_loc.reshape(1,self.n_cells)
            new_loc_temp = new_loc

            cell_tracks_list.append(list(new_loc[-1]))
            
        self.cell_tracks = np.array(cell_tracks_list).T