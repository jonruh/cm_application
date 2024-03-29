from scipy.integrate import odeint
import pandas as pd
import matplotlib as mpl
import numpy as np

class cc_cmodel_sol(object):

    def __init__(self, y0, t_eval, mmodel, cmodel, X, Fs_feed):
        self.cmodel = cmodel.__dict__
        self.mmodel = mmodel.__dict__
        self.init_vals = y0
        self.t_eval = t_eval
        self.X = X
        self.Fs_feed = Fs_feed
        self.sol = odeint(func=mmodel.cc_cmodel_odes,y0=y0,tfirst=True,t=t_eval,args=(cmodel,X,Fs_feed))

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
            self.sol_df[mmodel.species.iloc[index]['species'] + ' ' + mmodel.species.iloc[index]['unit']] = concentrations

        for index in mmodel.rates.index:
            rates = mmodel.cc_cmodel_odes(t=0, y=self.sol[-1], comp_mod=cmodel, X=X, Fs_feed=Fs_feed, returns='rates')[index::n_rates]
            self.sol_df[mmodel.rates.iloc[index]['rate'] + ' ' + mmodel.rates.iloc[index]['unit']] = rates

        # adding additional useful columns
        self.sol_df['comp_id'] = range(1,len(self.sol_df)+1)
        self.sol_df['liquid_vol [L]'] = (cmodel.CompVolumes.CompVol - cmodel.CompVolumes.CompVol*cmodel.GH.GH)*1000

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

                value = list(self.sol_df[solution])[compartment_id]
                color_norm = max(0,value-norm.vmin)/(norm.vmax-norm.vmin)

                pp = mpl.pyplot.Rectangle(xy=(Xmin,Ymin), width=Xmax-Xmin, height=Ymax-Ymin,linewidth=0,edgecolor='#000000',facecolor=cmap(color_norm))
                ax.add_patch(pp)

            CompMap = self.cmodel['CompMap']
            
            for zone in CompMap.Zone.unique():
                # find unique X_min-X_max pairs for a given zone
                unique_X = CompMap[CompMap['Zone'] == zone].drop_duplicates(subset=['X_min','X_max'])

                for i in range(0,len(unique_X)):
                    X_min = unique_X.iloc[i].X_min
                    X_max = unique_X.iloc[i].X_max
                    Y_min = CompMap[(CompMap['Zone'] == zone) & (CompMap['X_min'] == X_min) & (CompMap['X_max'] == X_max)].Y_min.min()
                    Y_max = CompMap[(CompMap['Zone'] == zone) & (CompMap['X_min'] == X_min) & (CompMap['X_max'] == X_max)].Y_max.max()

                    ax.plot([X_min,X_max],[Y_min,Y_min],color='black')
                    ax.plot([X_min,X_max],[Y_max,Y_max],color='black')

                # find unique Y_min-Y_max pairs for a given zone
                unique_Y = CompMap[CompMap['Zone'] == zone].drop_duplicates(subset=['Y_min','Y_max'])

                for i in range(0,len(unique_Y)):
                    Y_min = unique_Y.iloc[i].Y_min
                    Y_max = unique_Y.iloc[i].Y_max
                    X_min = CompMap[(CompMap['Zone'] == zone) & (CompMap['Y_min'] == Y_min) & (CompMap['Y_max'] == Y_max)].X_min.min()
                    X_max = CompMap[(CompMap['Zone'] == zone) & (CompMap['Y_min'] == Y_min) & (CompMap['Y_max'] == Y_max)].X_max.max()

                    ax.plot([X_min,X_min],[Y_min,Y_max],color='black')
                    ax.plot([X_max,X_max],[Y_min,Y_max],color='black')