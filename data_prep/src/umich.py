"""
umich
=====
re-BatteryData preparation of UMich pouch-cell expansion data.

Usage::

    python data_prep/src/umich.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): Partial_Regular/UMich_Pouch/data/<cell>/{cycling_wExpansion.csv, OCV_wExpansion.csv}.

Output (relative to ``OUT_ROOT``): UMich_Pouch/60secs_wunc/<id>.pkl and UMich_Pouch/60secs_wunc/metadata.csv.

Qnom = 5 Ah, Vmax = 4.2 V, Vmin = 3 V.
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

data_store_dir = os.path.join(OUT_ROOT, 'UMich_Pouch')
datalog_files = os.path.join(data_store_dir,'60secs_wunc')

metadata_csv = os.path.join(datalog_files, 'metadata.csv')

if not os.path.exists(datalog_files):
  os.makedirs(datalog_files)


    
umich = os.path.join(root_dir, os.path.join('UMich_Pouch','data'))
bats = os.listdir(umich)
bats = [os.path.join(umich,b) for b in bats if os.path.isdir(os.path.join(umich,b))]

cyc_data_paths = [os.path.join(b,'cycling_wExpansion.csv') for b in bats]
cap_data_paths = [os.path.join(b, 'OCV_wExpansion.csv') for b in bats]
    

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

Qnom = 5
Vmax = 4.2
Vmin = 3

for bp_id, bp in enumerate(cyc_data_paths[-1:]):
    
    data = pd.read_csv(bp)
    capdata = pd.read_csv(cap_data_paths[bp_id])
    
    V = data['Voltage [V]'].values
    I = data['Current [mA]'].values/1000
    Q = data['Capacity [Ah]'].values
    T = data['Temperature [C]'].values
    t = data['Time [s]'].values
    cycnum = data['Cycle number'].values
    
    Qch_ = Q*(I>=0)
    Qdch_ = Q *(I<0)
    
    #chg_ids = np.where(I >= 0)[0]
    #dchg_ids = np.where(I < 0)[0]

    #chg_breaks = np.where(np.diff(chg_ids) > 1)[0]
    #dchg_breaks = np.where(np.diff(dchg_ids) > 1)[0]

    #cstart = chg_ids[0]
    #dstart = dchg_ids[0]

    #pairs_chg = []
    #pairs_dchg = []

    #for c in chg_breaks:
    #    end = chg_ids[c] + 1
    #    pairs_chg.append((cstart, end))
    #    if end-cstart > 0:    
    #        Qch_[ cstart:end] = Qch_[cstart:end] - np.min(Qch_[cstart:end])
    #    cstart = end
    #Qch_[cstart:] = Qch_[cstart:] - np.min(Qch_[cstart:])
    #pairs_chg.append((cstart, len(Qch_)))

    #for d in dchg_breaks:
    #     end = dchg_ids[d] + 1
    #     pairs_dchg.append((dstart, end))
    #     if end-dstart > 0:
    #         Qdch_[ dstart:end] = Qdch_[dstart:end] - np.min(Qdch_[dstart:end])
    #     dstart = end
    #Qdch_[dstart:] = Qdch_[dstart:] - np.min(Qdch_[dstart:])
    #pairs_dchg.append((dstart, len(Qdch_)))
    
    Icap = capdata['Current [mA]'].values/1000
    Qcap =  capdata['Capacity [Ah]'].values
    ncyc_cap = capdata['Cycle number'].values
    tcap = capdata['Time [s]'].values
    
    uniq_cap_cycles = np.unique(ncyc_cap).astype(int).tolist()
    Qtraj = np.asarray([np.max(Qcap[np.where(ncyc_cap == c)[0]]) for c in uniq_cap_cycles])
    
    num_cycles = len(np.unique(cycnum))
    
    step = 0
    fec_count = 0
    FEC_data = []
    cnum_list = []
    Qfec = []

    while step < num_cycles:

        Vfec = []
        Ifec = []
        Qch_fec = []
        Qdch_fec = []
        Tfec = []
        tfec = []
        cnums = []

        Qth = 0

        while Qth < 2*Qnom and step < num_cycles:

            c = cycnum[step]
            Vstep = V[cycnum==c]
            Istep = I[cycnum==c]
            Qch_step = Qch_[cycnum==c]
            Qdch_step = Qdch_[cycnum==c]
            Tstep = T[cycnum==c]
            tstep = t[cycnum==c]
            cnum = cycnum[cycnum==c]

      
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
        #step_ids.append(step)

        Vfec = np.asarray(Vfec)
        Ifec = np.asarray(Ifec)
        Qch_fec = np.asarray(Qch_fec)
        Qdch_fec = np.asarray(Qdch_fec)
        Tfec = np.asarray(Tfec)
        tfec = np.asarray(tfec)
        cnums = np.asarray(cnums).astype(int)
    

        signal = np.column_stack((Vfec, Ifec, Qch_fec, Qdch_fec, Tfec, tfec))
        dsampled_signal = stat_downsampling(signal, chunk_sampling(tfec, 60), uncertainty = True)

        cycs_present = np.unique(cnums)
        #Qfec.append(np.median(Qcycs[cycs_present.astype(int)]))

        FEC_data.append(dsampled_signal)
        cnum_list.append(int(np.median(cnums)))

        print(dsampled_signal.shape)

    
    
    cnum_list = np.asarray(cnum_list)
    
    Qids = []
    fec_ids = []
    
    for cid, cn in enumerate(uniq_cap_cycles):
        closest_fec = np.argmin(np.abs(cnum_list - cn))
        Qids.append(Qtraj[cid])
        fec_ids.append(closest_fec)
        
    Qids = np.asarray(Qids)
    fec_ids = np.asarray(fec_ids)
    
    Qfec = capacity_interpolation(Qids, fec_ids, np.arange(len(FEC_data)))
    
    data_dict = {'cycle':FEC_data,
               'capacity':Qfec}

    with open(os.path.join(datalog_files, str(bp_id)+ '.pkl'), 'wb') as f:
        pickle.dump(data_dict, f)

    print(fec_count)
  

    bat_meta['Bat_id'].append(bp_id)
    bat_meta['Bat_name'].append(bp)
    bat_meta['Bat_path'].append(os.path.join(datalog_files, str(bp_id)+ '.pkl'))
    bat_meta['Bat_dataset'].append('UMich')
    bat_meta['Form Factor'].append('Pouch')
    bat_meta['Cathode'].append('NMC')
    bat_meta['Anode'].append('Graphite')
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
