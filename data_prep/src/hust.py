"""
hust
====
re-BatteryData preparation of HUST (Ma et al., 2022); stored under the dataset name 'Unknown'.

Usage::

    python data_prep/src/hust.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Cycling_full_regular/our_data/*.pkl.

Output (relative to ``OUT_ROOT``): Unknown/60secs/<id>.pkl and Unknown/60secs/metadata.csv.

Qnom = 1.1 Ah, Vmax = 3.6 V, Vmin = 2 V; LFP cathode, graphite anode, form factor
``Unknown`` (Table 1 of the paper).
The raw records are segmented into full equivalent cycles (FECs) of
``2 * Qnom`` absolute charge throughput and resampled to 60 s resolution
(``chunk_sampling`` + ``stat_downsampling``). Each battery is stored as a
pickle ``{'cycle': [X_c], 'capacity': Q}`` with one array ``X_c`` per FEC
whose first eight columns are ``[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]``.
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

from scipy.interpolate import CubicSpline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression

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
    new_signal[i,1] = np.sum( np.abs(Ichg[p][:-1]) * np.diff(t[p]) )/new_signal[i,-1] if (len(p)>1) or (new_signal[i,-1]>0) else np.median(Ichg[p])
    new_signal[i,2] = -np.sum( np.abs(Idchg[p][:-1]) * np.diff(t[p]) )/new_signal[i,-1] if (len(p)>1) or (new_signal[i,-1]>0) else np.median(Idchg[p])
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

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]

    return lmin,lmax

def split_cycles(cyc_data, num_cycs):

  """Same as ``data_prep.utils.split_cycles`` (not called by this script)."""
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

def capacity_interpolation(Qt, t, t_new, mode = 'Linear'):

    """Same as ``data_prep.utils.capacity_interpolation``."""
    if mode == 'Linear':
        qtraj_complete = np.interp(t_new, t, Qt)
    else:
        spl = CubicSpline(t, Qt)
        qtraj_complete = spl(t_new)

    return qtraj_complete

root = os.getcwd()
root_dir = os.path.join(root,'Cycling_full_regular')

data_store_dir = os.path.join(OUT_ROOT, 'Unknown')
datalog_files = os.path.join(data_store_dir,'60secs')

metadata_csv = os.path.join(datalog_files, 'metadata.csv')

if not os.path.exists(datalog_files):
  os.makedirs(datalog_files)

xjtu = os.path.join(root_dir, 'our_data')
bats = os.listdir(xjtu)
#bats = [b for b in bats if os.path.isdir(os.path.join(xjtu,b))]
#bat_paths = []

#for b in bats:
#  bpaths = os.listdir(os.path.join(xjtu,b))
#  for bp in bpaths:
#    bat_paths.append(os.path.join(os.path.join(xjtu,b),bp))

Qnom = 1.1
Vmax = 3.6
Vmin = 2

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

for bid, bat in enumerate(bats):

  with open(os.path.join(xjtu, bat), 'rb') as fp:
    data = pickle.load(fp)

  bdata = data[bat.replace('.pkl','')]
  qdata = bdata['dq']
  Qtraj = [qdata[k] for k in qdata.keys()]
  Qtraj = np.asarray(Qtraj)/1000

  data = bdata['data']
  num_cycles = len(Qtraj)

  step = 1
  step_ids = [step]
  fec_count = 0
  FEC_data = []

  while step <= num_cycles:

    Vfec = []
    Ifec = []
    Qcfec = []
    Qdfec = []
    Tfec = []
    tfec = []

    Qth = 0

    while Qth < 2*Qnom and step <= num_cycles:

      V = data[step]['Voltage (V)'].values
      I = data[step]['Current (mA)'].values /1000
      t = data[step]['Time (s)'].values
      tdiff = np.diff(t)
      tdiff = np.append(tdiff, tdiff[-1])
      T = np.asarray([25]*len(t)).astype(float)



      Ichg = I * (I>=0)
      Idchg = I*(I<0)

      Qc = np.cumsum(np.abs(Ichg) * tdiff/3600) * (I>=0)
      Qd = np.cumsum(np.abs(Idchg) * tdiff/3600) *(I<0)

      Qth += np.sum(np.abs(I)*tdiff/3600)

      Vfec.extend(V.tolist())
      Ifec.extend(I.tolist())
      Qcfec.extend(Qc.tolist())
      Qdfec.extend(Qd.tolist())
      Tfec.extend(T.tolist())
      if len(tfec) > 0:
        if tfec[-1] > t[0]:
          t = t + (tfec[-1] - t[0])

      tfec.extend(t.tolist())

      step += 1

    step_ids.append(step)

    Vfec = np.asarray(Vfec)
    Ifec = np.asarray(Ifec)
    Qcfec = np.asarray(Qcfec)
    Qdfec = np.asarray(Qdfec)
    Tfec = np.asarray(Tfec)
    tfec = np.asarray(tfec)

    signal = np.column_stack((Vfec, Ifec, Qcfec, Qdfec, Tfec, tfec))
    dsampled_signal = stat_downsampling(signal, chunk_sampling(tfec, 60), uncertainty = False)
    fec_count += 1
    FEC_data.append(dsampled_signal)
    print(dsampled_signal.shape)


  print(fec_count)
  Qfec = Qtraj[np.asarray(step_ids[:-1]).astype(int) - 1]
  plt.plot(Qfec)

  data_dict = {'cycle':FEC_data,
               'capacity':Qfec}

  with open(os.path.join(datalog_files, str(bid)+'.pkl'), 'wb') as fp:
    pickle.dump(data_dict, fp)

  with open(os.path.join(datalog_files, str(bid)+'.pkl'),'wb') as f:
      pickle.dump(data_dict, f)

  fec_inds = random.sample(np.arange(len(FEC_data)).tolist(), 16)

  fig, axs = plt.subplots(4,4, figsize = (20,10))
  for i in range(4):
    for j in range(4):

        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,4])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,3])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,2])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,1])

  bat_meta['Bat_id'].append(bid)
  bat_meta['Bat_path'].append(os.path.join(datalog_files, str(bid)+'.pkl'))
  bat_meta['Bat_name'].append(bat)
  bat_meta['Bat_dataset'].append('Unknown')
  bat_meta['Cathode'].append('LFP')
  bat_meta['Anode'].append('Graphite')
  bat_meta['Qnom'].append(Qnom)
  bat_meta['Vmax'].append(Vmax)
  bat_meta['Vmin'].append(Vmin)
  bat_meta['Form Factor'].append('Unknown')

metadata = pd.DataFrame(data = bat_meta,
                        columns = list(bat_meta.keys()),
                        index = None)

metadata.to_csv(metadata_csv)
