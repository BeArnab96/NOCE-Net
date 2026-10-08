"""
isu_ilcc
========
re-BatteryData preparation of ISU-ILCC (Iowa State University).

Usage::

    python data_prep/src/isu_ilcc.py --raw-root <RAW_ROOT> --out-root <OUT_ROOT>

Input (relative to ``RAW_ROOT``): valid_cells.csv, Cycling_json/Release {1.0,2.0}/<cell>.json and RPT_json/Release {1.0,2.0}/<cell>.json.

Output (relative to ``OUT_ROOT``): Iowa_Uni/60secs_wunc/<id>.pkl and Iowa_Uni/60secs_wunc/metadata.csv.

Qnom = 0.25 Ah, Vmax = 4.2 V, Vmin = 3 V.
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
import json
import numpy as np
import pandas as pd
from utils import stat_downsampling, capacity_interpolation, chunk_sampling
import matplotlib.pyplot as plt
import os, random, pickle

def convert_cycling_to_dict(cell,subfolder):
    """Load ``Cycling_json/<subfolder>/<cell>.json`` and convert its time stamps to ``np.datetime64``."""
    with open('Cycling_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))
        
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []
            
    for iii, start_time in enumerate(data_dict['time_series_charge']['start']):
        if start_time != '[]':
            data_dict['time_series_charge']['start'][iii] = np.datetime64(start_time)
            data_dict['time_series_charge']['stop'][iii] = np.datetime64(data_dict['time_series_charge']['stop'][iii])
            data_dict['time_series_discharge']['start'][iii] = np.datetime64(data_dict['time_series_discharge']['start'][iii])
            data_dict['time_series_discharge']['stop'][iii] = np.datetime64(data_dict['time_series_discharge']['stop'][iii])            
        else:
            data_dict['time_series_charge']['start'][iii] = []
            data_dict['time_series_charge']['stop'][iii] = []
            data_dict['time_series_discharge']['start'][iii] = []
            data_dict['time_series_discharge']['stop'][iii] = []
    
    for iii in range(len(data_dict['time_series_charge']['start'])):
        data_dict['QV_charge']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge']['t'][iii]))
        data_dict['QV_discharge']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge']['t'][iii]))
    
    return data_dict


def capacity_time_calculation(RPT_dict,cycling_dict):
    """Cumulative test time (weeks) at every reference performance test and the corresponding C/5 discharge capacity, as a DataFrame with columns ``Time`` and ``Capacity`` (missing values interpolated or dropped)."""
    char_time_sum_week = []
    cycling_time_sum_week = []
    
    for i,time_start in enumerate(RPT_dict['start_stop_time']['start']):
        if type(time_start) != list:
            time_start = np.datetime64(time_start)
            time_stop =  np.datetime64(RPT_dict['start_stop_time']['stop'][i])
            t = (time_stop-time_start)/np.timedelta64(1, 'W')
            char_time_sum_week.append(t)
        else:
            char_time_sum_week.append(np.nan)
        
    char_time_sum_week = np.array(char_time_sum_week)
        
    for i,data in enumerate(char_time_sum_week):
        if np.isnan(data)==True:
            if np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+1])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-1]+char_time_sum_week[i+1])/2
    
            elif np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+2])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-1]+char_time_sum_week[i+2])/2
    
            elif np.isnan(char_time_sum_week[i-2])==False and np.isnan(char_time_sum_week[i+1])==False:
                char_time_sum_week[i] = (char_time_sum_week[i-2]+char_time_sum_week[i+1])/2
    
    for i,time_start in enumerate(cycling_dict['start_stop_time']['start']):
        if type(time_start) != list:
            time_start = np.datetime64(time_start)
            time_stop =  np.datetime64(cycling_dict['start_stop_time']['stop'][i])
            t = (time_stop-time_start)/np.timedelta64(1, 'W')
            cycling_time_sum_week.append(t)
        else:
            cycling_time_sum_week.append(np.nan)
    
    cycling_time_sum_week = np.array(cycling_time_sum_week)    
    for i,data in enumerate(cycling_time_sum_week):
        if np.isnan(data)==True:
            if np.isnan(char_time_sum_week[i-1])==False and np.isnan(char_time_sum_week[i+1])==False:
                cycling_time_sum_week[i] = (cycling_time_sum_week[i-1]+cycling_time_sum_week[i+1])/2
    
    capacity_readings = RPT_dict['capacity_discharge_C_5']
    for i,c in enumerate(capacity_readings):
        if type(c)==list:
            capacity_readings[i]=np.nan      
    capacity_readings = np.array(capacity_readings)
    
    # Calculating the cumulative time on test each week
    cum_time = [0]
    for i,cycling_time in enumerate(cycling_time_sum_week):
        t = cum_time[-1]+char_time_sum_week[i]+cycling_time
        cum_time.append(t)
    cum_time = np.array(cum_time)

    capacity_fade_data = pd.DataFrame([cum_time,capacity_readings])
    capacity_fade_data = capacity_fade_data.T
    capacity_fade_data = capacity_fade_data.set_axis(['Time', 'Capacity'], axis=1)
    capacity_fade_data.dropna(inplace=True)
    
    return capacity_fade_data

def convert_RPT_to_dict(cell,subfolder):
    """Load ``RPT_json/<subfolder>/<cell>.json`` and convert its time stamps to ``np.datetime64``."""
    with open('RPT_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))
    
    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['start_stop_time']['start'])):
        data_dict['QV_charge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_2']['t'][iii]))
        data_dict['QV_discharge_C_2']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_2']['t'][iii]))
        data_dict['QV_charge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_charge_C_5']['t'][iii]))
        data_dict['QV_discharge_C_5']['t'][iii] = list(map(np.datetime64,data_dict['QV_discharge_C_5']['t'][iii]))
    
    return data_dict

if __name__ == '__main__':
    
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
    
    valid_cells = pd.read_csv('valid_cells.csv').values.flatten().tolist()
    batch2 = ['G57C1','G57C2','G57C3','G57C4','G58C1', 'G26C3','G49C1','G49C2','G49C3','G49C4','G50C1','G50C3','G50C4'] 
    Qnom = 0.25
    Vmax = 4.2
    Vmin = 3
    
    data_store_dir = os.path.join(OUT_ROOT, 'Iowa_Uni')
    datalog_files = os.path.join(data_store_dir,'60secs_wunc')
    metadata_csv = os.path.join(datalog_files, 'metadata.csv')
    
    if not os.path.exists(datalog_files):
        os.makedirs(datalog_files)
    
    for cell_id, cell in enumerate(valid_cells):
        
        if cell not in batch2:
            subfolder = 'Release 1.0'
        else:
            subfolder = 'Release 2.0'
        
        cycling_dict = convert_cycling_to_dict(cell,subfolder)
        RPT_dict = convert_RPT_to_dict(cell,subfolder)
        capacity_fade_data = capacity_time_calculation(RPT_dict,cycling_dict)
        
        
        chg_data = cycling_dict['QV_charge']
        chg_tstart, chg_tstop = cycling_dict['time_series_charge']['start'],cycling_dict['time_series_charge']['stop'] 
        dchg_data = cycling_dict['QV_discharge']
        dchg_tstart, dchg_tstop = cycling_dict['time_series_discharge']['start'],cycling_dict['time_series_discharge']['stop'] 
        
        assert len(chg_tstart) == len(chg_tstop)
        assert len(dchg_tstart) == len(dchg_tstop)
        assert len(chg_tstart) == len(dchg_tstart)
        
        num_steps = len(chg_tstart)
        
        #if chg_tstart[0] < dchg_tstart[0]:
            
        step = 0
        fec_count = 0
        step_counts = [step]
        date_count = [chg_tstart[0]]
        FEC_data = []
        
        while step < num_steps:
                
                Vcyc = []
                Icyc = []
                tcyc = []
                Tcyc = []
                Qccyc = []
                Qdcyc = []
                
                
                Qth = 0
                    
                while Qth < 2*Qnom and step < num_steps:
                        
                    V = list(chg_data['V'][step]) + list(dchg_data['V'][step])
                    I = list(chg_data['I'][step]) + list(dchg_data['I'][step])
                    tchg = np.asarray((chg_data['t'][step])).astype(float)/1e9
                    tdchg = np.asarray((dchg_data['t'][step])).astype(float)/1e9
                    #tdchg = ltdchg + tdchg 
                    t = list(tchg) + list(tdchg)
                    Qc = list(chg_data['Q'][step]) + [0] * len(dchg_data['I'][step])
                    Qd = [0] * len(chg_data['I'][step]) + list(dchg_data['Q'][step])
                    
                    
                    if len(t) > 0:
                        
                        tcdiff = np.diff(chg_data['t'][step]).astype(float)/1e9
                        tcdiff = np.append(tcdiff, tcdiff[-1])
                        
                        tddiff = np.diff(dchg_data['t'][step]).astype(float)/1e9
                        tddiff = np.append(tddiff, tddiff[-1])
                        
                        Qth += np.sum(np.abs(chg_data['I'][step]) * np.asarray(tcdiff)/3600) + np.sum(np.abs(dchg_data['I'][step]) * np.asarray(tddiff)/3600)
                        
                        Vcyc.extend(V)
                        Icyc.extend(I)
                        if len(tcyc) > 0:
                            if tcyc[-1] > t[0]:
                                t = list(np.asarray(t) + (tcyc[-1] - t[0]))
                        tcyc.extend(t)
                        Qccyc.extend(Qc)
                        Qdcyc.extend(Qd)
                        Tcyc.extend([30]*len(t))
                        
                        step+=1
                    
                    else:
                        step+=1
                        continue
                    
                if len(Vcyc) > 0:
                    
                    signal = np.column_stack((Vcyc, Icyc, Qccyc, Qdcyc, Tcyc, tcyc))
                    dsampled_signal = stat_downsampling(signal, chunk_sampling(signal[:,-1]), uncertainty = True)
                    
                    #print(dsampled_signal.shape)
                    fec_count += 1
                    step_counts.append(step)
                
                    if step<num_steps:
                        date_count.append(dchg_tstop[step])#-np.datetime64('1970-01-01T00:00:00Z')).astype(float)#/np.timedelta64(1, 's'))
                    else:
                        date_count.append(dchg_tstop[step-1])#-np.datetime64('1970-01-01T00:00:00Z')).astype(float)#/np.timedelta64(1, 's'))
                
                    FEC_data.append(dsampled_signal)
        
        print(len(FEC_data))
        
        if len(FEC_data)>50:
        
            fec_inds = random.sample(np.arange(len(FEC_data)).tolist(), 16)

            fig, axs = plt.subplots(4,4, figsize = (20,10))
            for i in range(4):
                for j in range(4):
              
                    axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,4])
                    axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,3])
                    axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,2])
                    axs[i,j].plot(FEC_data[fec_inds[int(i*4+j)]][:,1])
        
            cyc_date_count = np.array(date_count)
        #
            rpt_start_time = RPT_dict['start_stop_time']['start'][0]
        
            rpt_rel_time_week = capacity_fade_data['Time'].values
        
            rpt_real_time = np.array([(rpt_start_time + np.timedelta64(int(relt*7*24*60*60), 's')) for relt in rpt_rel_time_week])
        
            print(len(rpt_real_time), len(cyc_date_count))
        
            closest_cycs = np.zeros((len(rpt_real_time),))-1
            for rid, rt in enumerate(rpt_real_time):
                closest_cyc = np.argmin(np.abs((rt - cyc_date_count).astype(float)))
                if closest_cyc in closest_cycs:
                    pass
                else:
                    closest_cycs[rid] = closest_cyc
                
            Qcap = capacity_fade_data['Capacity']
            Qcap_val = Qcap[closest_cycs >= 0]
            cyc_ids = closest_cycs[closest_cycs >= 0]
        
            total_cyc_ids = np.arange(len(FEC_data))
            Qcap_interp = capacity_interpolation(Qcap_val, cyc_ids, total_cyc_ids)
        
            data_dict = {'cycle':FEC_data,
                     'capacity':Qcap_interp}
        
            bat_meta['Bat_id'].append(cell_id)
            bat_meta['Bat_path'].append(os.path.join(datalog_files, str(cell_id)+'.pkl'))
            bat_meta['Bat_name'].append(cell)
            bat_meta['Bat_dataset'].append('Iowa_Uni')
            bat_meta['Cathode'].append('NCM')
            bat_meta['Anode'].append('Graphite')
            bat_meta['Qnom'].append(Qnom)
            bat_meta['Vmax'].append(Vmax)
            bat_meta['Vmin'].append(Vmin)
            bat_meta['Form Factor'].append('Polymer')

            with open(os.path.join(datalog_files, str(cell_id)+'.pkl'),'wb') as f:
                pickle.dump(data_dict, f)
        
            plt.plot(Qcap_interp)
            
            print('Done', cell_id)
        
    metadata = pd.DataFrame(data = bat_meta,
                            columns = list(bat_meta.keys()),
                            index = None)

    metadata.to_csv(metadata_csv)
