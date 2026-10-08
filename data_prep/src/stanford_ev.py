"""
stanford_ev
===========
re-BatteryData preparation of Stanford-EV (Pozzato et al., 2022); dataset names 'Stanford_<cell>'.

Usage::

    python data_prep/src/stanford_ev.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Full_Irregular/Stanford/cycling_tests and Full_Irregular/Stanford/diagnostic_tests/capacity_test.mat.

Output (relative to ``OUT_ROOT``): Stanford/60secs_wunc/<id>.pkl and Stanford/60secs_wunc/metadata.csv.

Qnom = 4.85 Ah, Vmax = 4.2 V, Vmin = 2.5 V.
The cycling records of each cell are split into the cycles listed in the
per-cell tables ``bat_cyc_nums`` (``split_cycles``) and every cycle is
resampled to 60 s resolution (``chunk_sampling`` + ``stat_downsampling``).
Capacities from ``capacity_test.mat`` are interpolated to every cycle.
Each cell is stored as a pickle ``{'cycle': [X_c], 'capacity': Q}`` whose
arrays ``X_c`` have the columns ``[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]``
followed by four uncertainty columns.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import parse_paths  # noqa: E402

RAW_ROOT, OUT_ROOT = parse_paths(__doc__)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import os, re, random
from datetime import datetime
import natsort
import pickle
from scipy.io import loadmat
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from utils import stat_downsampling, chunk_sampling, hl_envelopes_idx, capacity_interpolation
import io


root = os.getcwd()
root_dir = os.path.join(root,os.path.join('Full_Irregular','Stanford'))

datalog_files = os.path.join(OUT_ROOT, 'Stanford', '60secs_wunc')
metadata_csv = os.path.join(datalog_files , 'metadata.csv')

if not os.path.exists(datalog_files):
  os.makedirs(datalog_files)
    


bat_cyc_nums = {}


bat_cyc_nums['W5'] = pd.read_csv(io.StringIO('''
25
50
50
34
8
20
7
25
25
25
25
25
25
25
'''), header=None).values[:,0].astype(int).tolist()



bat_cyc_nums['W8'] = pd.read_csv(io.StringIO('''
25
50
50
23
2
1
6
28
37
25
25
25
25
25
'''), header=None).values[:,0].astype(int).tolist()



bat_cyc_nums['W9'] =  pd.read_csv(io.StringIO('''
25
50
47
22
1
1
4
29
37
25
25
25
25
25
'''), header=None).values[:,0].astype(int).tolist()


bat_cyc_nums['W10'] =pd.read_csv(io.StringIO('''
25
50
47
24
2
3
8
29
37
25
25
25
25
25
'''), header=None).values[:,0].astype(int).tolist()


bat_cyc_nums['G1'] =pd.read_csv(io.StringIO('''
25
5
7
25
25
25
25
25
25
25
'''), header=None).values[:,0].astype(int).tolist()


bat_cyc_nums['V4'] =pd.read_csv(io.StringIO('''
20
25
25
25
25
25
25
24
25
25
'''), header=None).values[:,0].astype(int).tolist()

def split_cycles(cyc_data, num_cycs):

  """Same as ``data_prep.utils.split_cycles``."""
  data = []
  I = cyc_data['I_full_vec_M1_NMC25degC'][:,0].astype(float)
  V = cyc_data['V_full_vec_M1_NMC25degC'][:,0].astype(float)
  step = cyc_data['Step_Index_full_vec_M1_NMC25degC'][:,0].astype(int)
  Qc = cyc_data['ch_cap_full_vec_M1_NMC25degC'][:,0].astype(float)
  Qd = cyc_data['dis_cap_full_vec_M1_NMC25degC'][:,0].astype(float)
  t = cyc_data['t_full_vec_M1_NMC25degC'][:,0].astype(float)
  T = np.asarray([23]*len(t)).astype(float)

  step_diff = np.diff(step)
  cyc_ids = np.where(step_diff<0)[0]
  cyc_pairs = []
  start = 0
  for ci, c in enumerate(cyc_ids):
    end = c
    cyc_pairs.append((start, end))
    start = end
  cyc_pairs.append((start, len(I)))

  if len(cyc_pairs) != num_cycs:
    drive_cycle_ids = np.where(np.logical_and(Qc>0.3, Qd>0.3))[0]
    ids_diff = np.where(np.diff(drive_cycle_ids)>1)[0]
    cyc_ids = drive_cycle_ids[ids_diff] + 1
    cyc_pairs = []
    start = 0
    for ci, c in enumerate(cyc_ids):
      end = c
      cyc_pairs.append((start, end))
      start = end
    cyc_pairs.append((start, len(I)))
    print('Using alternative cycle finder')

  print(len(cyc_pairs), num_cycs)

  for i,c in enumerate(cyc_pairs):

    start = c[0]
    end = c[1]

    dat = np.column_stack((V[start:end], I[start:end], Qc[start:end], Qd[start:end], T[start:end], t[start:end]))

    #samples = importance_sampling(dat[:,0], dat[:,-1])  # Use voltage as to compute the sampling weights sampling signal
    #dat_dsampled = dat[samples,:]

    #dat_dsampled[:-1,-1] = np.diff(dat_dsampled[:,-1])
    #dat_dsampled[-1,-1] = dat_dsampled[-1,-2]

    #data.append(dat_dsampled)
    data.append(dat)

  #data = np.asarray(data)

  return data

cyc_dir = os.path.join(root_dir , 'cycling_tests')
diag_dir = os.path.join(root_dir , 'diagnostic_tests')

cyc_grps = os.listdir(cyc_dir)
bat_dir = {}
for cid, c in enumerate(cyc_grps):
  bats = os.listdir(os.path.join(cyc_dir,c))
  bats = [b for b in bats if b.endswith('.mat')]
  for b in bats:
    if b[:-4] in bat_dir.keys():
      bat_dir[b[:-4]].append(os.path.join(os.path.join(cyc_dir,c),b))
    else:
      bat_dir[b[:-4]] = [os.path.join(os.path.join(cyc_dir,c),b)]

for b in bat_dir.keys():
  bat_dir[b] = natsort.natsorted(bat_dir[b])


Qnom = 4.85
Vmax = 4.2
Vmin = 2.5

bat_keys = list(bat_dir.keys())

bat_meta = {'Bat_id':[],
            'Bat_path':[],
            'Bat_dataset':[],
            'Form Factor':[],
            'Cathode': [],
            'Anode':[],
            'Qnom':[],
            'Vmax':[],
            'Vmin':[]}

diag = loadmat(os.path.join(diag_dir ,'capacity_test.mat'))
Q = diag['cap']
row = diag['row_diag_number'][0].astype(int)
col = diag['col_cell_label'][0]
col = [c[0] for c in col]


# For statistical sampling - 60 secs

for k, key in enumerate(bat_keys): # Battery Names : G1, W4, etc.

  if False:
    pass
  else:
    print(k, key)
    bat_data_cycs = bat_dir[key]     # Battery file path for each cycling folder
    num_cycs_list = bat_cyc_nums[key]

    if len(num_cycs_list) != len(bat_data_cycs):
      min_len = min(len(num_cycs_list), len(bat_data_cycs))
      num_cycs_list = num_cycs_list[:min_len]
      bat_data_cycs = bat_data_cycs[:min_len]

    fec_count = 0
    dsampled_cycling = []
    fec_inits = [fec_count]

    for i, b in enumerate(bat_data_cycs):  # Opening bat file in a cycling folder and splitting it into FECs
      bat_data = loadmat(b)
      split_data = split_cycles(bat_data, num_cycs_list[i])

      for did, d in enumerate(split_data):
        dsampled_cycling.append(stat_downsampling(d, chunk_sampling(d[:,-1], time_bound = 60), uncertainty = True))
        
        fec_count+=1
      fec_inits.append(fec_count)

    #dsampled_cycling = np.concatenate(dsampled_cycling, axis = 0)

    print(len(dsampled_cycling), fec_count, len(fec_inits))


    fec_inits = np.asarray(fec_inits)
    Qtraj = np.zeros((len(row),))
    col_id = col.index(key)
    for i in range(len(row)):
      Qtraj[i] = np.max(Q[i,col_id])
   
    min_len = min(len(Qtraj), len(fec_inits))

    fc = fec_inits[:min_len]
    qt = Qtraj[:min_len]

    #spline_feats = SplineTransformer(degree=2, n_knots = 5)
    #ffeats = spline_feats.fit_transform(fc.reshape(-1,1))

    #lin_model = LinearRegression()
    #lin_model.fit(ffeats, qt.reshape(-1,1))

    qtraj_complete = np.interp(np.arange(len(dsampled_cycling)), fc, qt)
    #capacity_interpolation(qt, fc, np.arange(len(dsampled_cycling)))
    #lin_model.predict(spline_feats.transform(np.arange(len(dsampled_cycling)).reshape(-1,1))).reshape(-1,1)

    plt.plot(qtraj_complete)

    data_dict = {'cycle': dsampled_cycling,
               'capacity': qtraj_complete}

    with open( os.path.join(datalog_files , str(k)+'.pkl'),'wb') as f:
      pickle.dump(data_dict, f)
    
    fec_inds = random.sample(np.arange(len(dsampled_cycling)).tolist(), 16)

    fig, axs = plt.subplots(4,4, figsize = (20,10))
    for i in range(4):
      for j in range(4):
          
        axs[i,j].plot(dsampled_cycling[fec_inds[int(i*4+j)]][:,4])
        axs[i,j].plot(dsampled_cycling[fec_inds[int(i*4+j)]][:,3])
        axs[i,j].plot(dsampled_cycling[fec_inds[int(i*4+j)]][:,2])
        axs[i,j].plot(dsampled_cycling[fec_inds[int(i*4+j)]][:,1])
        #axs[i,j].set_title(filepath[0].replace(root, ''))


  bat_meta['Bat_id'].append(k)
  bat_meta['Bat_path'].append(os.path.join(datalog_files , str(k)+'.pkl'))
  bat_meta['Bat_dataset'].append('Stanford_' + key)
  bat_meta['Form Factor'].append('Cylindrical')
  bat_meta['Qnom'].append(Qnom)
  bat_meta['Vmax'].append(Vmax)
  bat_meta['Vmin'].append(Vmin)
  bat_meta['Anode'].append('Graphite/Silicon')
  bat_meta['Cathode'].append('NMC')



metadata = pd.DataFrame(data = bat_meta,
                        columns = list(bat_meta.keys()),
                        index = None)

metadata.to_csv(metadata_csv)
