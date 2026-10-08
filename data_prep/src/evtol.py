"""
evtol
=====
re-BatteryData preparation of evTOL (Bills et al., 2023).

Usage::

    python data_prep/src/evtol.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): evTOL/VAH*.csv.

Output (relative to ``OUT_ROOT``): evTOL/60secs_wunc/<id>.pkl and evTOL/60secs_wunc/metadata.csv.

Qnom = 3 Ah, Vmax = 4.2 V, Vmin = 2.5 V. Cells VAH09, VAH10, VAH11, VAH13 and VAH25-VAH28 are skipped.
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
from scipy.interpolate import CubicSpline
import os, re, random
from datetime import datetime
import natsort
import pickle
root = os.getcwd()
root_dir = os.path.join(root,'evTOL')

data_store_dir = os.path.join(OUT_ROOT, 'evTOL')
datalog_files = os.path.join(data_store_dir,'60secs_wunc')

metadata_csv = os.path.join(datalog_files, 'metadata.csv')

if not os.path.exists(datalog_files):
  os.makedirs(datalog_files)

bats_prob = [9, 10, 11, 13, 25, 26, 27, 28]

bats = [b for b in os.listdir(root_dir) if not b.endswith('impedance.csv') and b.startswith('VAH')]
valid_bats = natsort.natsorted([b for b in bats if int(re.findall(r'\d+', b)[0]) not in bats_prob])
print(valid_bats)

bat_paths = []

for b in valid_bats:
  bat_paths.append(os.path.join(root_dir,b))
  #for bp in bpaths:
  #  bat_paths.append(os.path.join(os.path.join(xjtu,b),bp))
print(bat_paths)

'''
VAH01.csv: Baseline
VAH02.csv: Extended cruise (1000 sec)
VAH05.csv: 10% power reduction during discharge (flight)
VAH06.csv: CC charge current reduced to C/2
VAH07.csv: CV charge voltage reduced to 4.0V
VAH09.csv: Thermal chamber temperature of 20 degrees C
VAH10.csv: Thermal chamber temperature of 30 degrees C
VAH11.csv: 20% power reduction during discharge (flight)
VAH12.csv: Short cruise length (400 sec)
VAH13.csv: Short cruise length (600 sec)
VAH15.csv: Extended cruise (1000 sec)
VAH16.csv: CC charge current reduced to 1.5C
VAH17.csv: Baseline
VAH20.csv: Charge current reduced to 1.5C
VAH22.csv: Extended cruise (1000 sec)
VAH23.csv: CV charge voltage reduced to 4.1V
VAH24.csv: CC charge current reduced to C/2
VAH25.csv: Thermal chamber temperature of 20 degrees C
VAH26.csv: Short cruise length (600 sec)
VAH27.csv: Baseline
VAH28.csv: 10% power reduction during discharge (flight)
VAH30.csv: Thermal chamber temperature of 35 degrees C'''

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

Qnom = 3
Vmax = 4.2
Vmin = 2.5

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

for bp_id, bp in enumerate(bat_paths):

  data = pd.read_csv(bp)
  V = data['Ecell_V'].values
  I = data['I_mA'].values/1000
  Qch = data['QCharge_mA_h'].values/1000
  Qdch = data['QDischarge_mA_h'].values/1000
  t = data['time_s'].values
  cyc_nums = data['cycleNumber'].values
  step = data['Ns'].values
  T = data['Temperature__C'].values



  tdiff = np.diff(t)
  tdiff = np.append(tdiff, tdiff[-1])
  Ich = I * (I >= 0)
  Idch = I * (I < 0)
  Qch_ = np.cumsum(np.abs(Ich) * tdiff/3600)
  Qdch_ = np.cumsum(np.abs(Idch) * tdiff/3600)

  chg_ids = np.where(I >= 0)[0]
  dchg_ids = np.where(I < 0)[0]

  chg_breaks = np.where(np.diff(chg_ids) > 1)[0]
  dchg_breaks = np.where(np.diff(dchg_ids) > 1)[0]

  cstart = chg_ids[0]
  dstart = dchg_ids[0]

  pairs_chg = []
  pairs_dchg = []

  for c in chg_breaks:
    end = chg_ids[c] + 1
    pairs_chg.append((cstart, end))
    if end-cstart > 0:

      Qch_[ cstart:end] = Qch_[cstart:end] - np.min(Qch_[cstart:end])
    cstart = end
  Qch_[cstart:] = Qch_[cstart:] - np.min(Qch_[cstart:])
  pairs_chg.append((cstart, len(Qch_)))

  for d in dchg_breaks:
    end = dchg_ids[d] + 1
    pairs_dchg.append((dstart, end))
    if end-dstart > 0:
      Qdch_[ dstart:end] = Qdch_[dstart:end] - np.min(Qdch_[dstart:end])
    dstart = end
  Qdch_[dstart:] = Qdch_[dstart:] - np.min(Qdch_[dstart:])
  pairs_dchg.append((dstart, len(Qdch_)))

  max_len = max(len(pairs_chg), len(pairs_dchg))
  cyc_pairs = pairs_chg if max_len == len(pairs_chg) else pairs_dchg


  Qcycs = np.asarray([max(np.max(Qch_[cyc_nums == c]), np.max(Qdch_[cyc_nums == c])) for c in np.unique(cyc_nums)])

  num_cycs = len(cyc_pairs)

  step = 0
  fec_count = 0
  FEC_data = []
  step_ids = [step]
  Qfec = []

  while step < num_cycs:

    Vfec = []
    Ifec = []
    Qch_fec = []
    Qdch_fec = []
    Tfec = []
    tfec = []
    cnums = []

    Qth = 0

    while Qth < 2*Qnom and step < num_cycs:

      start = cyc_pairs[step][0]
      end = cyc_pairs[step][1]

      Vstep = V[start:end]
      Istep = I[start:end]
      Qch_step = Qch_[start:end]
      Qdch_step = Qdch_[start:end]
      Tstep = T[start:end]
      tstep = t[start:end]
      cnum = cyc_nums[start:end]


      tdiff = np.diff(tstep)
      tdiff = np.append(tdiff, tdiff[-1])

      Qth += np.sum(np.abs(Istep) * tdiff/3600)

      Vfec.extend(Vstep.tolist())
      Ifec.extend(Istep.tolist())
      Qch_fec.extend(Qch_step.tolist())
      Qdch_fec.extend(Qdch_step.tolist())
      Tfec.extend(Tstep.tolist())
      cnums.extend(cnum.tolist())

      if len(tfec) > 0:
        if tfec[-1] > tstep[0]:
          tstep = tstep + tfec[-1] - tstep[0]
      tfec.extend(tstep)

      step+=1

    fec_count += 1
    step_ids.append(step)

    Vfec = np.asarray(Vfec)
    Ifec = np.asarray(Ifec)
    Qch_fec = np.asarray(Qch_fec)
    Qdch_fec = np.asarray(Qdch_fec)
    Tfec = np.asarray(Tfec)
    tfec = np.asarray(tfec)
    cnums = np.asarray(cnums)


    signal = np.column_stack((Vfec, Ifec, Qch_fec, Qdch_fec, Tfec, tfec))
    dsampled_signal = stat_downsampling(signal, chunk_sampling(tfec, 60), uncertainty = True)

    cycs_present = np.unique(cnums)
    Qfec.append(np.median(Qcycs[cycs_present.astype(int)]))

    FEC_data.append(dsampled_signal)

    print(dsampled_signal.shape)

  data_dict = {'cycle':FEC_data,
               'capacity':Qfec}

  with open(os.path.join(datalog_files, str(bp_id)+ '.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)

  print(fec_count)


  bat_meta['Bat_id'].append(bp_id)
  bat_meta['Bat_name'].append(bp)
  bat_meta['Bat_path'].append(os.path.join(datalog_files, str(bp_id)+ '.pkl'))
  bat_meta['Bat_dataset'].append('evTOL')
  bat_meta['Form Factor'].append('Cylindrical')
  bat_meta['Cathode'].append('Unknown')
  bat_meta['Anode'].append('Unknown')
  bat_meta['Qnom'].append(Qnom)
  bat_meta['Vmax'].append(Vmax)
  bat_meta['Vmin'].append(Vmin)

  fec_inds = random.sample(np.arange(len(FEC_data)).tolist(), 16)

  fig, axs = plt.subplots(4,4, figsize = (20,10))
  for i in range(4):
    for j in range(4):

        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,4])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,3])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,2])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,1])

  plt.plot(Qfec)

metadata = pd.DataFrame(data = bat_meta,
                        columns = list(bat_meta.keys()),
                        index = None)

metadata.to_csv(metadata_csv)
