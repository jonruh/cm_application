from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib as mpl

class sc_cmodel_sol(object):

    def __init__(self, y0, t_eval, mmodel, cmodel, X, Fs_feed):
        self.cmodel = cmodel.__dict__
        self.mmodel = mmodel.__dict__
        self.init_vals = y0
        self.t_eval = t_eval
        self.X = X
        self.Fs_feed = Fs_feed
        self.sol = odeint(func=mmodel.sc_cmodel_odes,y0=y0,tfirst=True,t=t_eval,args=(cmodel,X,Fs_feed))

        fig, ax = mpl.pyplot.subplots()
        ax.plot(self.sol)
        ax.set_xlabel('simulation timesteps')
        ax.set_ylabel('value')
        ax.set_title('overview of the simulation results \n (check if steady state is reached)')

        # create a solution dataframe
        self.sol_df = pd.DataFrame()
        n_species = len(mmodel.species)
        n_rates = len(mmodel.rates)

        for index in mmodel.species.index:
            concentrations = self.sol[-1][index::n_species]
            self.sol_df[mmodel.species.iloc[index]['species']] = concentrations

        for index in mmodel.rates.index:
            rates = mmodel.sc_cmodel_odes(t=0, y=self.sol[-1], cmodel=cmodel, X=X, Fs_feed=Fs_feed, returns='rates')[index::n_rates]
            self.sol_df[mmodel.rates.iloc[index]['rate']] = rates

        # adding additional useful columns
        self.sol_df['comp_id'] = range(1,len(self.sol_df)+1)
        self.sol_df['liquid_vol [L]'] = cmodel.VL

    def plot_solution(self):
        
        for solution in self.sol_df.columns[:-2]:

            cmap = mpl.cm.get_cmap('rainbow')
            # for colormap normalization
            vmin = 0
            vmax = np.round(self.sol_df[solution].max()*1.05,2)

            fig, ax = mpl.pyplot.subplots(figsize=(2.5, 11))
            ax.set_xlim(0,2)
            ax.set_ylim(0,11)
            ax.set_xlabel('radius (m)')
            ax.set_ylabel('height (m)')
            ax.set_title(solution)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax)

            for i in self.cmodel['CompMap'].index:
                Xmin = self.cmodel['CompMap'].iloc[i].X_min
                Xmax = self.cmodel['CompMap'].iloc[i].X_max
                Ymin = self.cmodel['CompMap'].iloc[i].Y_min
                Ymax = self.cmodel['CompMap'].iloc[i].Y_max

                compartment_id = int(self.cmodel['CompMap'].iloc[i].Zone-1)

                qS_rel = list(self.sol_df[solution])[compartment_id]
                color_norm = max(0,qS_rel-norm.vmin)/(norm.vmax-norm.vmin)

                pp = mpl.pyplot.Rectangle(xy=(Xmin,Ymin), width=Xmax-Xmin, height=Ymax-Ymin,linewidth=1,edgecolor='#000000',facecolor=cmap(color_norm))
                ax.add_patch(pp)