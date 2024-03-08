import pandas as pd

class cc_cmodel(object):

    def __init__(self,excel_path):
        self.FeedDist    = pd.read_excel(excel_path,sheet_name='FeedDist')
        self.CompFlows   = pd.read_excel(excel_path,sheet_name='CompFlows')
        self.CompVolumes = pd.read_excel(excel_path,sheet_name='CompVolumes')
        self.GH          = pd.read_excel(excel_path,sheet_name='GH')
        self.kla         = pd.read_excel(excel_path,sheet_name='kla')
        self.O_star      = pd.read_excel(excel_path,sheet_name='O_star')
        self.CompMap     = pd.read_excel(excel_path,sheet_name='CompMap')