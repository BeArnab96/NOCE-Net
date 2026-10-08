"""
matr
====
re-BatteryData preparation of MATR (Severson et al., 2019); stored under the dataset name 'Severson'.

Usage::

    python data_prep/src/matr.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Cycling_full_regular/Severson/ with the three batch .mat files; batch1.pkl, batch2.pkl and batch3.pkl are created there from the .mat files when missing.

Output (relative to ``OUT_ROOT``): Severson/60secs_wunc/<id>.pkl and Severson/60secs_wunc/metadata.csv.

Qnom = 1.1 Ah, Vmax = 3.6 V, Vmin = 2 V.
The raw records are segmented into full equivalent cycles (FECs) of
``2 * Qnom`` absolute charge throughput and resampled to 60 s resolution
(``chunk_sampling`` + ``stat_downsampling``). Each battery is stored as a
pickle ``{'cycle': [X_c], 'capacity': Q}`` with one array ``X_c`` per FEC
whose first eight columns are ``[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]`` (four uncertainty columns follow).
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
from scipy.interpolate import CubicSpline

import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle

root = './Cycling_full_regular/Severson/'
data_logs = os.path.join(OUT_ROOT, 'Severson', '60secs_wunc') + os.sep
if not os.path.exists(data_logs):
    os.makedirs(data_logs)

metadata_csv = data_logs + 'metadata.csv'

files = natsort.natsorted(os.listdir(root))

def build_batch_pickle(matFilename, prefix, out_pkl):
    """
    Convert one MATR batch file (``*_batchdata_updated_struct_errorcorrect*.mat``)
    into a pickle ``{<prefix><i>: {'cycle_life', 'charge_policy', 'summary', 'cycles'}}``.
    """
    f = h5py.File(matFilename)
    batch = f['batch']

    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
        cl = f[batch['cycle_life'][i,0]][0,0]
        policy = f[batch['policy_readable'][i,0]][:,0].tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
        summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
        summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
        summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
        summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
        summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
        summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
        summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                    summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                    'cycle': summary_CY}
        cycles = f[batch['cycles'][i,0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j,0]]))
            Qc = np.hstack((f[cycles['Qc'][j,0]]))
            Qd = np.hstack((f[cycles['Qd'][j,0]]))
            Qdlin = np.hstack((f[cycles['Qdlin'][j,0]]))
            T = np.hstack((f[cycles['T'][j,0]]))
            Tdlin = np.hstack((f[cycles['Tdlin'][j,0]]))
            V = np.hstack((f[cycles['V'][j,0]]))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j,0]]))
            t = np.hstack((f[cycles['t'][j,0]]))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}
            cycle_dict[str(j)] = cd

        cell_dict = {'cycle_life': cl, 'charge_policy':policy, 'summary': summary, 'cycles': cycle_dict}
        key = prefix + str(i)
        bat_dict[key]=   cell_dict

    with open(out_pkl, 'wb') as fp:
        pickle.dump(bat_dict, fp)


MATR_BATCHES = [('2017-05-12_batchdata_updated_struct_errorcorrect-004.mat', 'b1c', 'batch1.pkl'),
                ('2017-06-30_batchdata_updated_struct_errorcorrect.mat', 'b2c', 'batch2.pkl'),
                ('2018-04-12_batchdata_updated_struct_errorcorrect-006.mat', 'b3c', 'batch3.pkl')]

for mat_name, prefix, pkl_name in MATR_BATCHES:
    if not os.path.exists(root + pkl_name):
        build_batch_pickle(root + mat_name, prefix, root + pkl_name)

batch1 = pickle.load(open(root+'batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']

numBat1 = len(batch1.keys())
print(numBat1)

batch2 = pickle.load(open(root+'batch2.pkl','rb'))
# There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
# and put it with the correct cell from batch1
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482]

for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']

numBat2 = len(batch2.keys())
print(numBat2)

batch3 = pickle.load(open(root+'batch3.pkl','rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']

numBat3 = len(batch3.keys())
print(numBat3)

numBat = numBat1 + numBat2 + numBat3
print(numBat)

bat_dict = {**batch1, **batch2, **batch3}

def chunk_sampling(time_signal, time_bound = 60):

  """Same as ``data_prep.utils.chunk_sampling``."""
  tdiff = np.diff(time_signal)
  tdiff = np.append(0, tdiff)
  #tchunks = np.cumsum(tdiff)%time_bound
  chunk_steps = np.cumsum(tdiff)//time_bound
  chunks = np.unique(chunk_steps)
  #chunks = np.where(np.diff(tchunks)<0)[0]

  steps = []
  for c in chunks:
    steps.append(np.where(chunk_steps == c)[0])
  return steps

def stat_downsampling(signal, idxs, uncertainty = False):

  """Same as ``data_prep.utils.stat_downsampling``."""
  new_signal = np.empty((len(idxs), signal.shape[1]+2))
  input_uncertainty = np.empty((len(idxs), 3))
  out_uncertainty = np.empty((len(idxs), 1))

  I = signal[:,1]
  t = signal[:,-1]
  Ichg  = I * (I>=0)
  Idchg = I * (I<0)

  for i,p in enumerate(idxs):
    # V <- median
    # I : [Ichg, Idchg] and Qchg = np.sum(np.abs(Ichg[]))
    new_signal[i,-1] = signal[p[-1],-1]-signal[p[0],-1] if len(p)>1 else 0
    new_signal[i,0] = np.median(signal[p,0])
    new_signal[i,1] = np.sum( np.abs(Ichg[p][:-1]) * np.diff(t[p]) )/new_signal[i,-1] if len(p)>1 else np.median(Ichg[p])
    new_signal[i,2] = -np.sum( np.abs(Idchg[p][:-1]) * np.diff(t[p]) )/new_signal[i,-1] if len(p)>1 else np.median(Idchg[p])
    new_signal[i,3:5] = np.mean(signal[p,2:4], axis = 0)
    new_signal[i,5:7] = np.mean(signal[p,4:], axis = 0)


    if uncertainty:
      input_uncertainty[i,0] = np.sqrt(np.mean(np.square(Ichg[p] - new_signal[i,1])))
      input_uncertainty[i,1] = np.sqrt(np.mean(np.square(Idchg[p] - new_signal[i,2])))
      input_uncertainty[i,2] = np.sqrt(np.mean(np.square(signal[p, 4] - new_signal[i,5])))

      out_uncertainty[i,0] = np.median(np.abs(signal[p,0] - new_signal[i,0])) #median absolute deviation
      #lower_bound[i,:] = np.percentile(signal[p,:], 25, axis = 0)
      #upper_bound[i,:] = np.percentile(signal[p,:], 75, axis = 0)

  if uncertainty:
    return np.concatenate((new_signal, input_uncertainty, out_uncertainty), axis = -1)
  else:
    return new_signal

def capacity_interpolation(Qt, t, t_new):

    """Cubic-spline interpolation of capacities ``Qt`` measured at ``t`` to ``t_new``."""
    spl = CubicSpline(t, Qt)
    qtraj_complete = spl(t_new)

    return qtraj_complete

Qnom = 1.1
Vmax = 3.6
Vmin = 2

#batkeys = list(bat_dict.keys())
bat_meta = {'Bat_id':[],
            'Bat_name':[],
            'Bat_path':[],
            'Bat_dataset':[],
            'Form Factor':[],
            'Cathode': [],
            'Anode':[],
            'Qnom':[],
            'Vmax':[],
            'Vmin':[]}

numBats = 0
for bch in [1,2,3]:

  batch = pickle.load(open(root+'batch'+str(bch)+'.pkl','rb'))
  #batkeys = list(batch.keys())

  if bch == 1:

    del batch['b1c8']
    del batch['b1c10']
    del batch['b1c12']
    del batch['b1c13']
    del batch['b1c22']

    batch2 = pickle.load(open(root+'batch2.pkl','rb'))

    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    for i, bk in enumerate(batch1_keys):
      batch[bk]['cycle_life'] = batch[bk]['cycle_life'] + add_len[i]
      for j in batch[bk]['summary'].keys():
        if j == 'cycle':
            batch[bk]['summary'][j] = np.hstack((batch[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch[bk]['summary'][j])))
        else:
            batch[bk]['summary'][j] = np.hstack((batch[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
      last_cycle = len(batch[bk]['cycles'].keys())
      for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

  elif bch == 2:
    del batch['b2c7']
    del batch['b2c8']
    del batch['b2c9']
    del batch['b2c15']
    del batch['b2c16']

  elif bch == 3:

    del batch['b3c37']
    del batch['b3c2']
    del batch['b3c23']
    del batch['b3c32']
    del batch['b3c42']
    del batch['b3c43']

  bat_keys = list(batch.keys())
  #bat_keys = [b for b in bat_keys if b not in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']]

  for kid, key in enumerate(bat_keys):

    #if kid == 0:
    FEC_data = []
    bat = batch[key]
    clife = bat['cycle_life']
    Qs = bat['summary']['QD']


    cycles = bat['cycles']
    assert len(cycles.keys()) == len(Qs)
    cid= 0
    fec_count = 0
    cids = [cid]

    while cid < len(cycles.keys()):

      Icyc = []
      Vcyc = []
      Qccyc = []
      Qdcyc = []
      Tcyc = []
      tcyc =[]

      Qth = 0


      while Qth < 2*Qnom and cid < len(cycles.keys()):
        #for cid, cyc in enumerate(list(cycles.keys())):
          cyc = list(cycles.keys())[cid]
          cycle = cycles[cyc]
          I = cycle['I']
          Qc = cycle['Qc'] * (I>=0)
          Qd = cycle['Qd'] * (I<0)
          T = cycle['T']
          t = cycle['t']*60 # min to secs
          V = cycle['V']

          Qth += np.sum(np.abs(I)*np.append(np.diff(t), np.diff(t)[-1])/3600)
          Icyc.extend(I)
          Vcyc.extend(V)
          Qccyc.extend(Qc)
          Qdcyc.extend(Qd)
          Tcyc.extend(T)
          if len(tcyc)>0:
            if tcyc[-1] > t[0]:
              t = t + tcyc[-1] - t[0]

          tcyc.extend(t)

          cid += 1

          #print(key, cyc)
      print('Qratio:', Qth/(2*Qnom))

      signal = np.column_stack((np.asarray(Vcyc),
                                  np.asarray(Icyc),
                                  np.asarray(Qccyc),
                                  np.asarray(Qdcyc),
                                  np.asarray(Tcyc),
                                  np.asarray(tcyc)))
      dsampled_signal = stat_downsampling(signal, chunk_sampling(signal[:,-1], time_bound = 60), uncertainty= True)
      FEC_data.append(dsampled_signal)
      fec_count+=1
      cids.append(cid)
      print(dsampled_signal.shape)

    cids = [c for c in cids if c<len(Qs)]
    data_dict = {'cycle':FEC_data,
                 'capacity':Qs[np.asarray(cids).astype(int)]}
    with open(data_logs + str(numBats)+'.pkl', 'wb') as fp:
          pickle.dump(data_dict, fp)

    print('Num cycles:',len(FEC_data), 'for battery', numBats + kid)

    bat_meta['Bat_id'].append(numBats)
    bat_meta['Bat_name'].append(key)
    bat_meta['Bat_path'].append(data_logs + str(numBats)+'.pkl')
    bat_meta['Bat_dataset'].append('Severson')
    bat_meta['Form Factor'].append('Cylindrical')
    bat_meta['Cathode'].append('LFP')
    bat_meta['Anode'].append('Graphite')
    bat_meta['Qnom'].append(Qnom)
    bat_meta['Vmax'].append(Vmax)
    bat_meta['Vmin'].append(Vmin)

    numBats += 1
  #print(numBats)
df = pd.DataFrame(bat_meta, columns = list(bat_meta.keys()), index = None)
df.to_csv(metadata_csv)
