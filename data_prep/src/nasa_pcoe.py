"""
nasa_pcoe
=========
re-BatteryData preparation of NASA PCoE battery aging data.

Usage::

    python data_prep/src/nasa_pcoe.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Cycling_full_regular/NASA_PCoE/<group>/B00xx.mat.

Output (relative to ``OUT_ROOT``): NASA_PCoE/60secs/<id>.pkl and NASA_PCoE/60secs/metadata.csv.

Qnom = 2 Ah, Vmax = 4.2 V; Vmin is set per cell.
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
from glob import glob
import dateutil.parser
import re

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

def split_cycles(cyc_data, chg_pairs, dchg_pairs):
  """Concatenate the charge segment ``chg_pairs[i]`` and the discharge segment ``dchg_pairs[i]`` of the record ``cyc_data`` (rows V, I, Qc, Qd, T, ..., t) into cycle ``i``; returns a list of (n_i, 6) arrays ``[V, I, Qc, Qd, T, t]``."""
  data = []
  I = cyc_data[1]
  V = cyc_data[0]
  Qc = cyc_data[2]
  Qd = cyc_data[3]
  t = cyc_data[-1]
  T = cyc_data[4]

  chg_ids = chg_pairs
  dchg_ids = dchg_pairs

  for i, c in enumerate(chg_ids):
    chg_data = np.column_stack((V[c[0]:c[1]], I[c[0]:c[1]], Qc[c[0]:c[1]], Qd[c[0]:c[1]], T[c[0]:c[1]], t[c[0]:c[1]]))
    if i < len(dchg_ids):
      d = dchg_ids[i]
      dchg_data = np.column_stack((V[d[0]:d[1]], I[d[0]:d[1]], Qc[d[0]:d[1]], Qd[d[0]:d[1]], T[d[0]:d[1]], t[d[0]:d[1]]))

      data.append(np.concatenate((chg_data, dchg_data), axis = 0))
    else:
      data[-1] = np.concatenate((data[-1], chg_data), axis = 0)
  return data

parent_dir = './Cycling_full_regular/NASA_PCoE/'
dir_list = natsort.natsorted([d for d in os.listdir(parent_dir) if os.path.isdir(parent_dir + d)])
print(dir_list)
data_logs = os.path.join(OUT_ROOT, 'NASA_PCoE', '60secs') + os.sep
metadata_csv = data_logs + 'metadata.csv'

if not os.path.exists(data_logs):
  os.makedirs(data_logs)

filepath = []
cellname = []
for i in range(len(dir_list)):

  cell_data_files = os.listdir(parent_dir+dir_list[i])
  for f in cell_data_files:
    if not f.endswith('.mat') or f.endswith('.txt'):
      cell_data_files.remove(f)

  for c in cell_data_files:
    filepath.append(parent_dir+dir_list[i]+'/' + c)
    cellname.append(c[:-4])
print(len(filepath))
print(filepath[0])

Qnom = 2
Vmax = 4.2

invalid_files = ['./NASA_PCoE/1. BatteryAgingARC-FY08Q4/B0018.mat',
                 './NASA_PCoE/2. BatteryAgingARC_25_26_27_28_P1/B0026.mat',
                 './NASA_PCoE/2. BatteryAgingARC_25_26_27_28_P1/B0027.mat',
                 './NASA_PCoE/2. BatteryAgingARC_25_26_27_28_P1/B0028.mat',
                 './NASA_PCoE/2. BatteryAgingARC_25_26_27_28_P1/B0025.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0028.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0026.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0027.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0025.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0034.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0036.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0033.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0039.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0038.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0040.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0042.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0041.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0043.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0029.mat',
                 './NASA_PCoE/3. BatteryAgingARC_25-44/B0044.mat',
                 './NASA_PCoE/5. BatteryAgingARC_49_50_51_52/B0050.mat',
                 './NASA_PCoE/5. BatteryAgingARC_49_50_51_52/B0052.mat',
                 './NASA_PCoE/5. BatteryAgingARC_49_50_51_52/B0051.mat',
                 './NASA_PCoE/5. BatteryAgingARC_49_50_51_52/B0049.mat',
                 './NASA_PCoE/6. BatteryAgingARC_53_54_55_56/B0053.mat',
                 './NASA_PCoE/6. BatteryAgingARC_53_54_55_56/B0055.mat',
                 './NASA_PCoE/6. BatteryAgingARC_53_54_55_56/B0056.mat',
                 './NASA_PCoE/6. BatteryAgingARC_53_54_55_56/B0054.mat',
                 ]

for f in invalid_files:

    cell = f[-9:-4]
    datanum = int(f[12])
    data = loadmat(parent_dir + f[12:])[cell]
    data = data[0,0][0]
    num_cycles = data.shape[1]

    data_trimmed = []
    ref_inds = []

    for c in range(num_cycles):
      op_type = data[0,c][0][0]
      if 'charge' in op_type:
        data_trimmed.append(data[0,c])


    for step in range(len(data_trimmed)):

      Vstep = data_trimmed[step][3][0,0][0][0]
      Istep = data_trimmed[step][3][0,0][1][0]
      Tstep = data_trimmed[step][3][0,0][2][0]
      tstep = data_trimmed[step][3][0,0][5][0]

      combo = np.column_stack((Vstep, Istep, Tstep, tstep))

      if np.any(np.isnan(combo)):
        print('Nan at step '+str(step)+' for cell no ' + cell+' at data num '+str(datanum))

filepath = [(f,i) for i,f in enumerate(filepath) if f.replace('/Cycling_full_regular','') not in invalid_files]
cellnames = [cellname[filepath[k][1]] for k in range(len(filepath))]
filepath = [filepath[k][0] for k in range(len(filepath))]

filecell = [(f, cellnames[i]) for i,f in enumerate(filepath) if not f.endswith('.txt')]
filepath = [f[0] for f in filecell]
cellname = [f[1] for f in filecell]

Vmin = dict.fromkeys(cellname)
Vmin['B0006'] = 2.5
Vmin['B0007'] = 2.2
Vmin['B0005'] = 2.7
Vmin['B0032'] = 2.7
Vmin['B0031'] = 2.5
Vmin['B0030'] = 2.2
Vmin['B0045'] = 2
Vmin['B0046'] = 2.2
Vmin['B0047'] = 2.5
Vmin['B0048'] = 2.7

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

for filenum, filename in enumerate(filepath):

  print(filename)

  if filename.replace('/Cycling_full_regular','') in invalid_files:
    print('Got Invalid File')
    pass

  else:

    data = loadmat(filename)[cellname[filenum]]
    data = data[0,0][0]

    # Collect all the RW datapoints and reference cycles. Low current cycles for OCV-SOC id and remaining are discarded

    num_cycles = data.shape[1]
    data_trimmed = []
    data_chg = []
    data_dchg = []
    ref_inds = []
    cdate = []
    ddate = []
    dates = []

    for c in range(num_cycles):
      op_type = data[0,c][0][0]
      if 'charge' in op_type:
        data_chg.append(data[0,c])

        year, month, day, hour, minute, second = tuple(data[0,c][2][0].astype(int).tolist())

        cdate.append(datetime(year, month, day, hour, minute, second))
        dates.append(datetime(year, month, day, hour, minute, second))
        data_trimmed.append(data[0,c])

      if 'discharge' in op_type:
        data_dchg.append(data[0,c])
        year, month, day, hour, minute, second = tuple(data[0,c][2][0].astype(int).tolist())
        ddate.append(datetime(year, month, day, hour, minute, second))
        dates.append(datetime(year, month, day, hour, minute, second))
        ref_inds.append(c)
        data_trimmed.append(data[0,c])

    #data_trimmed = align(data_chg, data_dchg, cdate, ddate)
    date_diff = np.asarray([(dates[i] - dates[i-1]).total_seconds() for i in range(1, len(dates))])
    flag = np.prod(date_diff>=0)
    if flag == 0:
      print('Cycles not aligned with dates')


    # Break into full equivalent cycles, sample and compute SOC

    FEC_data = []

    step = 0
    fec_count = 0

    while step < len(data_trimmed):

      qfinal = 0

      Vfec = []
      Ifec = []
      #qcfec = []
      #qdfec = []
      Tfec = []
      tfec = []
      stepfec = []

      stepcount = 0

      while qfinal < 2*Qnom and step < len(data_trimmed):

        #if len(data_trimmed[step][4][0]) <= 1:
        #  step += 1
        #  continue

        op_type = data_trimmed[step][0][0]
        Vstep = data_trimmed[step][3][0,0][0][0]
        Istep = data_trimmed[step][3][0,0][1][0]
        Tstep = data_trimmed[step][3][0,0][2][0]
        tstep = data_trimmed[step][3][0,0][5][0]


        tstep_diff = (tstep[1:] - tstep[:-1]).tolist()
        tstep_diff.append(tstep_diff[-1])
        tstep_diff = np.asarray(tstep_diff)

        qchg = np.cumsum(np.abs(Istep)*(tstep_diff/3600))

        #qc = qc - np.min(qc)
        #qd = qd - np.min(qd)

        Vfec.extend(Vstep.tolist())
        Ifec.extend(Istep.tolist())
        #qcfec.extend(qc.tolist())
        #qdfec.extend(qd.tolist())
        Tfec.extend(Tstep.tolist())
        if len(tfec)>0:
          if tstep[0] < tfec[-1]:
              tstep = tstep + (tfec[-1] - tstep[0] + tfec[-1] - tfec[-2])
        tfec.extend(tstep.tolist())
        stepfec.extend(list((np.ones((len(tstep),))*step).astype(int)))

        #qfec = qfec + qchg.tolist()
        #stepfec = stepfec + list((np.ones((len(rtstep),))*step).astype(int))

        #assert(len(socfec) == len(Vfec))

        #if np.max(socfec) > 1.2:
        #  socfec = (np.asarray(socfec) - np.max(socfec) + 1).tolist()

        qfinal += qchg[-1]
        step += 1
        stepcount += 1
        if step % 1000 == 0:
          print(step)
        #print(step)

      Ifec = np.asarray(Ifec)
      tfec = np.asarray(tfec)
      tstep_diff = tfec[1:] - tfec[:-1]
      tstep_diff = np.append(tstep_diff, tstep_diff[-1])

      qc = np.zeros(len(Ifec))
      qd = np.zeros(len(Ifec))

      Ichg = Ifec * (Ifec>=0)
      Idchg = Ifec * (Ifec<0)

      chg_ids = np.where(Ifec>=0)[0]
      chg_breaks = np.where(np.diff(chg_ids)>1)[0]
      chg_break_ids = chg_ids[chg_breaks] + 1

      dchg_ids = np.where(Ifec<0)[0]
      dchg_breaks = np.where(np.diff(dchg_ids)>1)[0]
      dchg_break_ids = dchg_ids[dchg_breaks] + 1

      if len(chg_ids)>1:
        start = chg_ids[0]
        for i in range(len(chg_break_ids)):
            end = chg_break_ids[i]
            qc[start:end] = np.cumsum(np.abs(Ichg[start:end])*tstep_diff[start:end]/3600)
            start = end
        end = chg_ids[-1]
        qc[start:end] = np.cumsum(np.abs(Ichg[start:end])*tstep_diff[start:end]/3600)

      if len(dchg_ids)>1:
        start = dchg_ids[0]

        for i in range(len(dchg_break_ids)):
            end = dchg_break_ids[i]
            qd[start:end] = np.cumsum(np.abs(Idchg[start:end])*tstep_diff[start:end]/3600)
            start = end
        end = dchg_ids[-1]
        qd[start:end] = np.cumsum(np.abs(Idchg[start:end])*tstep_diff[start:end]/3600)

      #if len(Vfec) > 200:
      #  samples = importance_sampling(np.asarray(Ifec), np.asarray(tfec))
      #else:
      #  samples = np.arange(len(Vfec))

      Vfec = np.asarray(Vfec)
      qcfec = np.asarray(qc)
      qdfec = np.asarray(qd)
      Tfec = np.asarray(Tfec)

      #rtfec = tfec
      #rtfec = rtfec[1:] - rtfec[:-1]
      #rtfec = np.append(rtfec, rtfec[-1])

      #fecsamp = np.ones((len(Vfec),))*fec_count
      stepsamp = np.asarray(stepfec)

      fec_count += 1

      fec_data = np.column_stack((Vfec, Ifec, qcfec, qdfec, Tfec, tfec))

      idxs = chunk_sampling(tfec, time_bound = 60)
      ssamp = []
      for i,p in enumerate(idxs):
        ssamp.append(np.median(stepsamp[p]))

      ssamp = np.asarray(ssamp).astype(int).reshape(-1,1)

      dsampled_data = np.concatenate((stat_downsampling(fec_data, idxs, uncertainty = False), ssamp), axis = -1)

      FEC_data.append(dsampled_data)

    #FEC_data = np.asarray(FEC_data)
    print(len(FEC_data), fec_count)

    FEC_data_stored = [f[:,:-1] for f in FEC_data]

    fec_inds = random.sample(np.arange(len(FEC_data)).tolist(), 16)

    fig, axs = plt.subplots(4,4, figsize = (20,10))
    for i in range(4):
      for j in range(4):

        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,4])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,3])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,2])
        axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,1])
        axs[i,j].set_title(str(fec_inds[int(i*4+j)]))

    fec_nans = np.any(np.asarray([np.any(np.isnan(f)) for f in FEC_data]))
    if fec_nans:
      print('Nan in data '+ str(step)+' '+filepath[filenum])
      break

    # Find the reference capacity curve
    Qref = []
    for cind in range(len(ref_inds)):
      qd = data_trimmed[cind][3][0,0][-1][0][0]
      Istep = data_trimmed[cind][3][0,0][1][0]
      tstep = data_trimmed[cind][3][0,0][5][0]
      tstep_diff = (tstep[1:] - tstep[:-1]).tolist()
      tstep_diff.append(tstep_diff[-1])
      tstep_diff = np.asarray(tstep_diff)
      qd_alt = np.sum(np.abs(Istep)*tstep_diff/3600)

      Qref.append(max(qd, qd_alt))

    ref_inds = np.asarray(ref_inds)
    Qref = np.asarray(Qref)
    print(Qref.shape)

    Qref_ind = []
    Qref_fec = []
    for i in range(len(FEC_data)):

      steps = FEC_data[i][:,-1]
      step_unique = np.unique(steps)
      refs = []
      for u in step_unique:
        refs.append(ref_inds[np.argmin(np.abs(ref_inds-u))])

      ref_uniq, votes = np.unique(refs, return_counts=True)
      Qref_ind.append(ref_uniq[np.argmax(votes)])
      Qref_fec.append(Qref[np.where(ref_inds==Qref_ind[-1])[0]])

    Qref_fec = np.asarray(Qref_fec)

    plt.plot(Qref_fec)

    data_dict = {'cycle':FEC_data_stored,
               'capacity':Qref_fec}

    with open(os.path.join(data_logs,str(filenum)+'.pkl'), 'wb') as f:
      pickle.dump(data_dict, f)


    bat_meta['Bat_id'].append(filenum)
    bat_meta['Bat_name'].append(filename)
    bat_meta['Bat_path'].append(os.path.join(data_logs,str(filenum)+'.pkl'))
    bat_meta['Bat_dataset'].append('NASA PCoE')
    bat_meta['Cathode'].append('NMC')
    bat_meta['Qnom'].append(Qnom)
    bat_meta['Vmax'].append(Vmax)
    bat_meta['Vmin'].append(Vmin[cellname[filenum]])
    bat_meta['Anode'].append('Graphite')
    bat_meta['Form Factor'].append('Cylindrical')

metadata = pd.DataFrame(data = bat_meta,
                        columns = list(bat_meta.keys()),
                        index = None)

metadata.to_csv(metadata_csv)
