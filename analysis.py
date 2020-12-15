import pandas as pd

vti = pd.read_csv('VTI.csv')
bnd = pd.read_csv('BND.csv')
blv = pd.read_csv('BLV.csv')
vti['Date'] = pd.to_datetime(vti['Date'],format='%Y-%m-%d')
bnd['Date'] = pd.to_datetime(bnd['Date'],format='%Y-%m-%d')
blv['Date'] = pd.to_datetime(blv['Date'],format='%Y-%m-%d')





