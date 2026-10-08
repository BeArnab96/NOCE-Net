"""
xjtu_partial
============
re-BatteryData preparation of XJTU partial-cycle data (Wang et al., 2024).

Usage::

    python data_prep/src/xjtu_partial.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Partial_Regular/XJTU/<batch>/*.mat.

Output (relative to ``OUT_ROOT``): XJTU_Partial/60secs_wunc/<id>.pkl and XJTU_Partial/metadata.csv.

Qnom = 2 Ah, Vmax = 4.2 V, Vmin = 2.5 V.
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
from scipy.io import loadmat
from utils import stat_downsampling, chunk_sampling, hl_envelopes_idx, capacity_interpolation

root = os.getcwd()
root_dir = os.path.join(root,'Partial_Regular')

data_store_dir = os.path.join(OUT_ROOT, 'XJTU_Partial')
datalog_files = os.path.join(data_store_dir,'60secs_wunc')

metadata_csv = os.path.join(data_store_dir, 'metadata.csv')

if not os.path.exists(datalog_files):
  os.makedirs(datalog_files)


    
xjtu = os.path.join(root_dir, 'XJTU')
bats = os.listdir(xjtu)
bats = [b for b in bats if os.path.isdir(os.path.join(xjtu,b))]
bat_paths = []

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

Qnom = 2
Vmax = 4.2
Vmin = 2.5

for b in bats:
  bpaths = os.listdir(os.path.join(xjtu,b))
  for bp in bpaths:
    bat_paths.append(os.path.join(os.path.join(xjtu,b),bp))


very_high_Q_cycles = []


for bp_id, bp in enumerate(bat_paths):

    dataset = loadmat(bp)
    data = dataset['data']
    summary = dataset['summary']

    num_cycles = summary[0][0][1].shape[0]
    Qtraj = summary[0][0][1]

    #Vcyc = []
    #Qcyc = []
    #tcyc = []
    #Tcyc = []
    #Icyc = []
    #fec_cyc = []
    cyc_feats = []
    fec_count = 0
    fec_dist = []

    n = 0

    while n < num_cycles:

      Qth_ratio = 0
      Qth_ratio_plus1 = 0

      #time = []
      tacc = []
      Qacc = []
      Vacc = []
      Iacc = []
      Tacc = []
      tdiff_acc = []
      ndist = []

      while Qth_ratio  < 2*Qnom and n < num_cycles:

        time_arr = data[0][n][0]

        #print(time[0][0][0])
        time = []

        for tid, t  in enumerate(time_arr):
          s = t[0][0]

          try:
            time.append(datetime.strptime(s, '%Y-%m-%d,%H:%M:%S'))
          except:
            if isinstance(s[0], float):
              if len(time)>2:
                #print(s)
                past_1 = time[-1]
                past_2 = time[-2]
                time.append(past_1 + (past_1 - past_2))
              elif not isinstance(time_arr[tid+1][0][0][0], float):
                #print(s)
                time.append(datetime.strptime(time_arr[tid+1][0][0], '%Y-%m-%d,%H:%M:%S'))
              else:
                k = 1
                while isinstance(time_arr[tid+k][0][0][0], float):
                  k+=1
                time.append(datetime.strptime(time_arr[tid+k][0][0], '%Y-%m-%d,%H:%M:%S'))

                #time.append(time[-1])



        #time = [datetime.strptime(t[0][0], '%Y-%m-%d,%H:%M:%S') for t in time]
        tdiff = [(time[i]-time[i-1]).total_seconds() for i in range(1,len(time))]

        tdiff.append(tdiff[-1])

        V = data[0][n][2][:,0]
        I = data[0][n][3][:,0]
        T = data[0][n][6][:,0]
        Q = data[0][n][4][:,0]

        Qth_ratio = Qth_ratio + np.sum(np.abs(I[:len(tdiff)]) * np.asarray(tdiff)) / (3600)


        if Qth_ratio < 8:
          Qacc.extend(Q.tolist())
          Vacc.extend(V.tolist())
          Iacc.extend(I.tolist())
          Tacc.extend(T.tolist())
          #tdiff_acc.extend(tdiff)
          tacc.extend(time)
          ndist.append(n)
          n+=1

        elif len(ndist) < 1:
          Qacc.extend(Q.tolist())
          Vacc.extend(V.tolist())
          Iacc.extend(I.tolist())
          Tacc.extend(T.tolist())
          #plt.plot(Q)
          #tdiff_acc.extend(tdiff)
          tacc.extend(time)
          ndist.append(n)
          n+=1
          print(Qth_ratio, len(ndist), n, n-1)
          very_high_Q_cycles.append([Q.tolist(), V.tolist(), I.tolist()])

        else:
          Qth_ratio = Qth_ratio - np.sum(np.abs(I) * np.asarray(tdiff)) / (3600)
          break

        #if n+1 < num_cycles:

        #  tn_1 = data[0][n+1][0]
        #  In_1 = data[0][n+1][3][:,0]
        #  tn_1 = [datetime.strptime(t[0][0], '%Y-%m-%d,%H:%M:%S') for t in tn_1]
        #  tn_1_diff = [(tn_1[i]-tn_1[i-1]).seconds for i in range(1,len(tn_1))]
        #  tn_1_diff.append(tn_1_diff[-1])

        #  Qth_ratio_plus1 = np.sum(np.abs(In_1) * np.asarray(tn_1_diff)) / (3600*2)

        #else:
        #  Qth_ratio_plus1 = 0

      fec_dist.append(np.asarray(ndist).astype(int))

      time_fec = np.asarray(tacc)
      tdiff_fec = [(time_fec[i]-time_fec[i-1]).total_seconds() for i in range(1,len(time_fec))]
      tdiff_fec = np.append(0, np.asarray(tdiff_fec))
      t_norm = np.cumsum(tdiff_fec)
      
      
      signal_acc = np.column_stack((np.asarray(Vacc), 
                                    np.asarray(Iacc), 
                                    np.asarray(Qacc)*(np.asarray(Iacc)>=0), 
                                    np.asarray(Qacc)*(np.asarray(Iacc)<0), 
                                    np.asarray(Tacc), 
                                    t_norm))
      
      dsampled_data = stat_downsampling(signal_acc, 
                                        chunk_sampling(t_norm, 
                                                       time_bound = 60), 
                                        uncertainty = True)
      cyc_feats.append(dsampled_data)
      print(dsampled_data.shape)
      #time_fec = time_fec[sampled_ids]
      #tdiff_fec = [(time_fec[i]-time_fec[i-1]).seconds for i in range(1,len(time_fec))]
      #tdiff_fec.append(tdiff_fec[-1])

      #Vcyc.append(np.asarray(Vacc)[sampled_ids].reshape(-1,1))
      #Icyc.append(np.asarray(Iacc)[sampled_ids].reshape(-1,1))
      #Qcyc.append(np.asarray(Qacc)[sampled_ids].reshape(-1,1))
      #Tcyc.append(np.asarray(Tacc)[sampled_ids].reshape(-1,1))
      #tcyc.append(np.asarray(tdiff_fec).reshape(-1,1))
      #fec_cyc.append(np.asarray([fec_count]*len(sampled_ids)).reshape(-1,1))


      fec_count += 1

    print(fec_count)

    #Vcyc = np.asarray(Vcyc)
    #Icyc = np.asarray(Icyc)
    #Tcyc = np.asarray(Tcyc)
    #Qcyc = np.asarray(Qcyc)
    #tcyc = np.asarray(tcyc)
    #fec_cyc = np.asarray(fec_cyc).astype(int)

    #cyc_feats = np.concatenate((Vcyc, Icyc, Tcyc, Qcyc, tcyc, fec_cyc), axis = -1)
    print(bp_id, len(cyc_feats))

    # Enveloping Q if needed

    lmin, lmax = hl_envelopes_idx(Qtraj[:,0])
    qtraj = median_filter(Qtraj[lmax][:,0],5)
    Qup = Qtraj[:,0][lmax]
    
    Qtraj = capacity_interpolation(Qup, lmax, np.arange(Qtraj.shape[0]).astype(int))
    
    #Qtraj_copy = np.zeros_like(Qtraj)

    #for i, l in enumerate(lmax):
    #  if i == 0:
    #    Qtraj_copy[:l,0] = qtraj[i] - (qtraj[i+1] - qtraj[i])/(lmax[i+1]-l)*np.arange(l)
    #  elif i+1<len(lmax):
    #    Qtraj_copy[l:lmax[i+1],0] = qtraj[i]+((qtraj[i+1]-qtraj[i])/l)*np.arange(lmax[i+1]-l)
    #  else:
    #    Qtraj_copy[l:,0] = qtraj[i] + (qtraj[i] - qtraj[i-1])/(l- lmax[i-1])*np.arange(Qtraj.shape[0]-l)

    #if np.mean(np.abs(Qtraj_copy - Qtraj)/(Qtraj_copy+1e-3)) > 0.5:
    #  Qtraj = Qtraj_copy
    
    Qfec = np.empty((fec_count,))

    for fid, f in enumerate(fec_dist):
      Qfec[fid] = np.median(Qtraj[f])

    plt.plot(Qfec)

    data_dict = {'cycle':cyc_feats,
                 'capacity':np.asarray(Qfec)}

    bat_meta['Bat_id'].append(bp_id)
    bat_meta['Bat_path'].append(os.path.join(datalog_files, str(bp_id)+'.pkl'))
    bat_meta['Bat_name'].append(bp)
    bat_meta['Bat_dataset'].append('XJTU_Partial')
    bat_meta['Cathode'].append('NCM')
    bat_meta['Anode'].append('Graphite')
    bat_meta['Qnom'].append(2)
    bat_meta['Vmax'].append(4.2)
    bat_meta['Vmin'].append(2.5)
    bat_meta['Form Factor'].append('Cylindrical')

    with open(os.path.join(datalog_files, str(bp_id)+'.pkl'),'wb') as f:
      pickle.dump(data_dict, f)
      
    fec_inds = random.sample(np.arange(len(cyc_feats)).tolist(), 16)

    fig, axs = plt.subplots(4,4, figsize = (20,10))
    for i in range(4):
      for j in range(4):
          
        axs[i,j].plot(cyc_feats[fec_inds[int(i*4+j)]][:,4])
        axs[i,j].plot(cyc_feats[fec_inds[int(i*4+j)]][:,3])
        axs[i,j].plot(cyc_feats[fec_inds[int(i*4+j)]][:,2])
        axs[i,j].plot(cyc_feats[fec_inds[int(i*4+j)]][:,1])



metadata = pd.DataFrame(data = bat_meta,
                        columns = list(bat_meta.keys()),
                        index = None)

metadata.to_csv(metadata_csv)
