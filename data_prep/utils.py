"""
data_prep.utils
===============
Resampling helpers shared by the dataset preparation scripts in
``data_prep/src``.

Raw signals are arrays with the columns ``[V, I, Qc, Qd, T, t]`` (voltage,
signed current, charge and discharge throughput, temperature, time in s).
"""

import numpy as np
from scipy.interpolate import CubicSpline


def chunk_sampling(time_signal, time_bound = 60):
  """
  Group consecutive samples into windows of ``time_bound`` seconds.

  Args:
      time_signal: (n,) time stamps in s.
      time_bound: window length in s.

  Returns:
      list of index arrays, one per non-empty window.
  """

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
  """
  Summarise each window of a raw signal by statistics.

  For every index array ``p`` in ``idxs`` one row
  ``[V, Ichg, Idchg, Qc, Qd, T, t, dt]`` is produced: the median voltage,
  the time-averaged charge and discharge currents, the mean throughputs,
  temperature and time, and the window duration ``dt``.

  Args:
      signal: (n, 6) raw signal ``[V, I, Qc, Qd, T, t]``.
      idxs: output of :func:`chunk_sampling`.
      uncertainty: also return the RMS deviation of ``Ichg``, ``Idchg`` and
          ``T`` and the median absolute deviation of ``V`` within each window.

  Returns:
      (len(idxs), 8) array, or (len(idxs), 12) with ``uncertainty=True``.
  """

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

def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Indices of the lower and upper envelopes of a 1-d signal.

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
  """
  Split the full-record arrays of one Stanford-EV cell
  (``*_full_vec_M1_NMC25degC`` fields) into cycles at decreases of the step
  index (or, if this does not give ``num_cycs`` cycles, at gaps between
  samples with ``Qc > 0.3`` and ``Qd > 0.3``). Temperature is set to 23 C.

  Returns:
      list of (n_c, 6) arrays ``[V, I, Qc, Qd, T, t]``.
  """

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
    """Interpolate capacities ``Qt`` measured at ``t`` to ``t_new`` (linear or cubic spline)."""
    
    if mode == 'Linear':
        qtraj_complete = np.interp(t_new, t, Qt)
    else:
        spl = CubicSpline(t, Qt)
        qtraj_complete = spl(t_new)
    
    return qtraj_complete