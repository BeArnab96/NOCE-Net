"""
noce_net.dataset
================
Data loading for NOCE-Net in the ``re-BatteryData`` format.

Expected folder layout (produced by ``data_prep/assemble_dataset.py``)::

    <data_root>/
        Train/
            train_cycle_numel.npy
            <DatasetFolder>/metadata.csv
            <DatasetFolder>/60secs/<battery_id>.pkl
            ...
        Test/
            test_cycle_numel.npy
            <DatasetFolder>/...

Each battery pickle holds ``{'cycle': [X_1, ..., X_N], 'capacity': Q}``
where ``X_c`` is the (L_c, 8) array of FEC ``c`` with columns
``[V, Ichg, Idchg, Qc, Qd, Temp, t, dt]`` resampled to 60 s and ``Q`` is
the capacity recorded at the start of each FEC.

Notation used throughout the code base
--------------------------------------
* ``B``   : number of batteries in a batch
* ``L``   : number of FECs (outer sequence length) in a batch or chunk
* ``T``   : intra-cycle length, i.e. time steps per FEC after padding
* ``d_u`` : input features per time step (``din`` = 7:
  Ichg, Idchg, Qc, Qd, Temp, t, dt)
* ``eta`` : task vector (6,) = [form factor, anode, cathode, Q_nom, V_max, V_min]
* ``gamma_c`` : FEC feature (2,) = [FEC index c, FEC duration Delta_c in hours]

A single sample is the full lifetime of one battery, returned as the
tuple documented in :meth:`Battery_Dataset.get_sample`.
"""
import gc
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import medfilt
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


def importance_sampling_indices(time_series: np.ndarray, M: int, max_neighbors: int = 5) -> np.ndarray:
    """
    Perform importance sampling on a 1D time series array and return M unique, sorted indices.

    Sampling probability is proportional to ``|x'| + 50 |x''|`` (first and
    second numerical derivatives of the series), so that segments with
    rapid changes are kept at higher resolution.

    Args:
        time_series (np.ndarray): 1D time series data.
        M (int): Number of unique indices to return.
        max_neighbors (int): Max number of neighbors to include around each sampled index.

    Returns:
        np.ndarray: Sorted array of unique sampled indices.
    """
    N = len(time_series)

    # Compute first and second derivatives
    first_derivative = np.gradient(time_series)
    second_derivative = np.gradient(first_derivative)
    importance_scores = np.abs(first_derivative) + 50*np.abs(second_derivative)

    # Normalize to get probability distribution
    prob_dist = importance_scores / np.sum(importance_scores)

    # Initial sampling (with possible repetitions)
    sampled_indices = np.random.choice(np.arange(N), size=M, p=prob_dist)

    # Count occurrences of each index
    counts = Counter(sampled_indices)

    # Expand indices by adding neighbors based on count (limited by max_neighbors)
    unique_indices = set()
    for idx, count in counts.items():
        # Number of neighbors to add = min(count, max_neighbors)
        half_window = min(count, max_neighbors)
        for offset in range(-half_window, half_window + 1):
            neighbor_idx = idx + offset
            if 0 <= neighbor_idx < N:
                unique_indices.add(neighbor_idx)

    # Ensure exactly M unique indices by trimming or padding
    unique_indices = sorted(unique_indices)
    if len(unique_indices) > M:
        unique_indices = sorted(np.random.choice(unique_indices, size=M, replace=False))
    elif len(unique_indices) < M:
        remaining = sorted(set(np.arange(N)) - set(unique_indices))
        pad = sorted(np.random.choice(remaining, size=M - len(unique_indices), replace=False))
        unique_indices = sorted(unique_indices + pad)

    return np.array(unique_indices)

class Battery_Dataset(Dataset):
  """
  Battery-level dataset over one split folder (``Train`` or ``Test``).

  The metadata of every dataset sub-folder is concatenated (in
  ``os.listdir`` order). With ``cap_mode='joint'`` and ``mode`` in
  ``{'train', 'val'}``, the batteries of every source dataset are split
  80/20 (in metadata order) into ``train_ids`` and ``val_ids``; the
  ``'val'`` part is the in-sample (Seen) evaluation set. With
  ``mode='test'`` all batteries are exposed through ``data_ids``.

  ``__getitem__`` receives a *list* of indices (as yielded by
  :class:`Battery_Batch_Sampler`) and returns a list of samples, which
  :func:`collate_batch` pads into tensors.

  Args:
      bat_path: path to the split folder (``<data_root>/Train`` or ``<data_root>/Test``).
      mode: ``'train'``, ``'val'`` or ``'test'``.
      cap_mode: ``'joint'`` for the standard multi-dataset setting, or the
          name of a single source dataset (``Bat_dataset`` value, or
          ``'Stanford'``) to restrict the dataset to it.
      num_shot: kept for interface compatibility; not used.
  """

  def __init__(self, bat_path, mode = 'train', cap_mode = 'joint', num_shot = 1):

    super().__init__()
    self.bat_path = bat_path

    bats = os.listdir(bat_path)
    self.bat_paths = [os.path.join(self.bat_path,b) for b in bats if os.path.isdir(os.path.join(self.bat_path,b))]
    self.meta_paths = [os.path.join(b, 'metadata.csv') for b in self.bat_paths]
    self.full_metadata = pd.concat([pd.read_csv(m) for m in self.meta_paths])
    self.full_metadata.reset_index(drop = True, inplace = True)

    self.mode = mode
    self.cap_mode = cap_mode

    self.bat_datasets = self.full_metadata['Bat_dataset'].values
    unique_datasets, inv_inds, data_count = np.unique(self.bat_datasets, return_counts = True, return_inverse = True)

    if cap_mode == 'joint':

      data_count_full = data_count[inv_inds]
      weights = 1/data_count_full
      self.weights = weights/np.max(weights)

      if self.mode in ['train', 'val']:
        self.train_ids, self.val_ids = [], []
        for u in unique_datasets:
          g = np.where(self.bat_datasets == u)[0].astype(int)
          self.train_ids.extend(g[:int(0.8*len(g))].tolist())
          self.val_ids.extend(g[int(0.8*len(g)):].tolist())
      else:
        self.data_ids = [i for i in range(len(self.full_metadata))]

    else:

      if cap_mode in unique_datasets:
        self.full_metadata = self.full_metadata[self.full_metadata['Bat_dataset'] == cap_mode]
      elif cap_mode == 'Stanford':
        ids = [i for i in range(len(self.full_metadata)) if str(self.bat_datasets[i]).startswith('Stanford')]
        self.full_metadata = self.full_metadata.iloc[ids]
      else:
        print('Wrong battery name!!')
        return

      metadata_ids = np.arange(len(self.full_metadata))
      np.random.shuffle(metadata_ids)
      self.data_ids = metadata_ids
      self.weights = [1]*len(metadata_ids)

  def get_task_vector(self, id):
    """
    Build the task vector ``eta`` of the battery at metadata row ``id``.

    Returns:
        np.ndarray of shape (6,): categorical codes for form factor, anode
        and cathode, followed by ``Qnom`` [Ah], ``Vmax`` [V] and ``Vmin`` [V].
    """

    task_dict = {'Form_factor':{'Cylindrical':0,
                                'Prismatic':1,
                                'Pouch':2,
                                'Polymer':3},

                 'Anode':{'Graphite':0,
                          'Graphite/Silicon':1,
                          'Other':2,
                          'Unknown':3},

                 'Cathode':{'LFP':0,
                            'NMC':1,
                            'NCM':1,
                            'NCA':2,
                            'LCO':3,
                            'NCM_NCA':4,
                            'Unknown':5}}

    task_vector = np.zeros((6,))

    metadata = self.full_metadata.iloc[id]

    task_vector[0] = task_dict['Form_factor'][metadata['Form Factor']]
    task_vector[1] = task_dict['Anode'][metadata['Anode']]
    task_vector[2] = task_dict['Cathode'][metadata['Cathode']]
    task_vector[3] = metadata['Qnom']
    task_vector[4] = metadata['Vmax']
    task_vector[5] = metadata['Vmin']

    return task_vector

  def __len__(self):
    """Number of batteries exposed by this dataset for the selected ``mode``."""

    if self.cap_mode == 'joint':
      if self.mode == 'train':
        return len(self.train_ids)
      elif self.mode == 'val':
        return len(self.val_ids)
      else:
        return len(self.full_metadata)
    else:
      return len(self.full_metadata)

  def get_sample(self, id):
    """
    Load and normalise the full lifetime of one battery.

    Pre-processing applied on the fly:

    * FECs longer than 700 steps are reduced to 700 steps (uniform
      re-binning below 3500 steps, importance sampling above), averaging
      continuous features and summing accumulated ones.
    * Voltage is min-max scaled with ``(V - Vmin) / (Vmax - Vmin)``;
      currents and throughputs are divided by ``Qnom``; temperature by 25;
      time is rescaled to [0, 1] within each FEC; ``dt`` is divided by 60.
    * Capacity is converted to normalised SOH
      ``(Q / Qnom - 0.8) / 0.2``, smoothed per dataset by
      :meth:`process_cap` and divided by its maximum.

    Args:
        id: position within ``train_ids`` / ``val_ids`` / ``data_ids``
            (depending on ``mode`` and ``cap_mode``).

    Returns:
        Tuple ``(inp, out, cap, eta, index, gamma, mask, weight, dataset_name)``:

        * ``inp``   : (N, T, d_u) inputs, columns [Ichg, Idchg, Qc, Qd, Temp, t, dt]
        * ``out``   : (N, T, 1) normalised voltage (target)
        * ``cap``   : (N, 1) normalised SOH trajectory
        * ``eta``   : (6,) task vector
        * ``index`` : (N, T) intra-cycle time index (1-based, 0 = padding)
        * ``gamma`` : (N, 2) FEC features [c, Delta_c (h)]
        * ``mask``  : (N, T) 1 for valid time steps, 0 for padding
        * ``weight``: inverse-frequency weight of the source dataset
        * ``dataset_name`` : value of the ``Bat_dataset`` metadata column
    """

    if self.cap_mode == 'joint':
      if self.mode == 'train':
        idx = self.train_ids[int(id)]
      elif self.mode == 'val':
        idx = self.val_ids[int(id)]
      else:
        idx = self.data_ids[int(id)]
    else:
      idx = self.data_ids[int(id)]

    mdata = self.full_metadata.iloc[idx]
    try:
      bpath_ = [mdata['Bat_path'].split('/')[-3], mdata['Bat_path'].split('/')[-1]]
    except:
      bpath_ = [mdata['Bat_path'].split('\\')[-3], mdata['Bat_path'].split('\\')[-1]]

    dataset_name = mdata['Bat_dataset']

    if dataset_name == 'NASA_Rand':
      dataset_num = int(bpath_[1].split('.')[0])
    else:
      dataset_num = None

    bpath = os.path.join(bpath_[0], '60secs', bpath_[1])

    datapath = os.path.join(self.bat_path, bpath)
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    num_cycs = len(data['cycle'])

    cyc_lens = [data['cycle'][c].shape[-2] for c in range(num_cycs)]
    where_cycs_high = np.where(np.asarray(cyc_lens)>700)[0]

    if len(where_cycs_high) >= 1:

      for h in range(len(where_cycs_high)):
        #high_idx = where_cycs_high[h]

        if data['cycle'][where_cycs_high[h]].shape[-2] < 3500:
          uni_samples = np.linspace(0, data['cycle'][where_cycs_high[h]].shape[-2], 700).astype(int)
        else:
          uni_samples = importance_sampling_indices(np.sum(data['cycle'][where_cycs_high[h]][:,1:3], axis = -1) , 700, 100)

        new_cyc_sample = np.zeros((700, data['cycle'][where_cycs_high[0]].shape[-1]))

        for i in range(699):
          new_cyc_sample[i,:3] = np.mean(data['cycle'][where_cycs_high[h]][uni_samples[i]:uni_samples[i+1],:3], axis = 0)
          new_cyc_sample[i,3:5] = np.sum(data['cycle'][where_cycs_high[h]][uni_samples[i]:uni_samples[i+1],3:5], axis = 0)
          new_cyc_sample[i,5:7] = np.mean(data['cycle'][where_cycs_high[h]][uni_samples[i]:uni_samples[i+1],5:7], axis = 0)
          new_cyc_sample[i,7] = np.sum(data['cycle'][where_cycs_high[h]][uni_samples[i]:uni_samples[i+1],7])

        data['cycle'][where_cycs_high[h]] = new_cyc_sample

    new_cyc_lens = [data['cycle'][c].shape[-2] for c in range(num_cycs)]
    max_clen = np.max(new_cyc_lens)

    num_feats = data['cycle'][0].shape[-1]

    inpdata = torch.empty((num_cycs, max_clen, num_feats-1))
    outdata = torch.empty((num_cycs, max_clen, 1))
    cycle_mask = torch.zeros((num_cycs, max_clen))
    index_data = torch.zeros((num_cycs, max_clen))
    fec_index = torch.arange(1, num_cycs+1) # 0 is for padded entries
    fec_cyclens = torch.zeros((num_cycs,))

    task_data = self.get_task_vector(idx)

    for c in range(num_cycs):

      if data['cycle'][c].shape[0] == 1:
        data['cycle'][c] = data['cycle'][c].squeeze(0)

      assert len(data['cycle'][c].shape) == 2


      data['cycle'][c] = torch.from_numpy(data['cycle'][c])#.reshape(1, data['cycle'][c].shape[0], -1)

      if torch.isnan(data['cycle'][c]).any():
          data['cycle'][c] = torch.nan_to_num(data['cycle'][c])

      fec_cyclens[c] = (torch.max(data['cycle'][c][:,6]) - torch.min(data['cycle'][c][:,6]))/3600

      data['cycle'][c][:,0] = (data['cycle'][c][:,0] - task_data[-1])/(task_data[-2]-task_data[-1])
      data['cycle'][c][:,1:5] = data['cycle'][c][:,1:5]/task_data[-3]
      data['cycle'][c][:,5] = data['cycle'][c][:,5]/25
      data['cycle'][c][:,6] = (data['cycle'][c][:,6] - torch.min(data['cycle'][c][:,6]))/(3600*fec_cyclens[c])
      data['cycle'][c][:,7] = data['cycle'][c][:,7]/60

      #cyc_len = data['cycle'][c].shape[0]
      pad_len = max_clen - data['cycle'][c].shape[0]
      padded_data = F.pad(data['cycle'][c].permute(1,0), (0,pad_len)).permute(1,0)

      inpdata[c] = padded_data[:,1:]
      outdata[c] = padded_data[:,0:1]
      cycle_mask[c, :data['cycle'][c].shape[0]] = 1
      index_data[c, :data['cycle'][c].shape[0]] = torch.arange(1, data['cycle'][c].shape[0]+1)  # 0 is for padded entries. Actual entries start with 1


    try:
      caps = data['capacity']
    except:
      caps = np.asarray(data['capacity'])

    if dataset_name == 'UMich':
      capdata = np.zeros_like(caps) + np.nan
    elif dataset_name == 'NASA Rand':
      if dataset_num in [0,1,2]:
        capdata = np.zeros_like(caps) + np.nan
      else:
        capdata = caps
    else:
      capdata = caps

    Qnom = task_data[-3]

    capdata = capdata/Qnom

    if bpath.startswith('Unknown'):
      capdata = capdata/1000
    capdata = (capdata - 0.8)/0.2 # Normalization: 1 - Nominal Capacity, 0 - First EOL
    #capdata[capdata<0] = 0
    cap_cleaned = self.process_cap(capdata, dataset_name, dataset_num)
    #capdata = cap_cleaned
    cap_cleaned = cap_cleaned /(np.max(cap_cleaned))
    capdata = torch.from_numpy(cap_cleaned).reshape(-1,1)

    #print(f'Input data shape: {inpdata.shape}, Out data shape: {outdata.shape}, Capacity Data shape: {capdata.shape}, Mask shape: {cycle_mask.shape}' )

    assert torch.sum(cycle_mask) >= torch.count_nonzero(outdata)

    fec_total = torch.column_stack((fec_index, fec_cyclens))

    return (inpdata, outdata, capdata, task_data, index_data, fec_total, cycle_mask, self.weights[idx], dataset_name)


  def __getitem__(self, indices):
        """Return ``[get_sample(i) for i in indices]`` (one batch of batteries)."""

        batch = []
        for bat_id in indices:
          sample = self.get_sample(bat_id)
          batch.append(sample)
          gc.collect()
        return batch

  def process_cap(self, capdata, dataset_name, dataset_num = None):
    """
    Dataset-specific cleaning of a normalised capacity trajectory.

    Outliers are replaced by local means (MATR/``'Severson'``), spikes are
    removed with median/maximum filters (TJU/``'NatComm'``,
    XJTU-Partial, HUST/``'Unknown'``) and noisy series are smoothed with
    moving averages (NASA-Rand, XJTU-Full, NASA-PCoE). Other datasets are
    returned unchanged.

    Args:
        capdata: (N,) or (N, 1) normalised capacity trajectory.
        dataset_name: ``Bat_dataset`` value of the battery.
        dataset_num: battery number (used only for NASA-Rand).

    Returns:
        np.ndarray of shape (N,) with the cleaned trajectory.
    """

    capdata = capdata.reshape(-1)

    if dataset_name == 'Severson':

      high_ids = np.where(capdata>1.5)[0]
      low_ids = np.where(capdata< -0.5)[0]
      all_anomal_ids = high_ids.tolist() + low_ids.tolist()

      if len(all_anomal_ids) > 0:

        for i in all_anomal_ids:

          smoothing_ids_low = []
          smoothing_ids_high = []

          j = 1
          count = 0
          while i - j >= 0 and count < 25:
            if i-j not in all_anomal_ids:
              smoothing_ids_low.append(i-j)
              count += 1
            j += 1

          j = 1
          count = 0
          while i + j < len(capdata) and count < 25:
            if i+j not in all_anomal_ids:
              smoothing_ids_high.append(i+j)
              count += 1
            j+=1

          smoothing_ids = smoothing_ids_high + smoothing_ids_low
          capdata_smoothers = capdata[smoothing_ids]
          capdata[i] = np.mean(capdata_smoothers)

      cap_updated = median_filter(capdata, 10)

    elif dataset_name == 'NatComm':
      median_trend = medfilt(capdata, kernel_size=5)
      threshold = 0.5  # or define based on noise estimate
      cap_updated = np.where((capdata < threshold * median_trend) + (capdata > threshold * median_trend) , median_trend, capdata)

    elif dataset_name == 'XJTU_Partial':
      window_size = 5  # tune this based on spike width
      cap_updated = maximum_filter1d(capdata, size=window_size)

    elif dataset_name == 'NASA Rand':
      if dataset_num not in [0,1,2]:
        w = 30
        cap_updated = np.convolve(capdata.reshape(-1), np.ones(w)*1/w, mode = 'same')
      else:
        cap_updated = capdata

    elif dataset_name == 'XJTU_Full':
       w = 30
       cap_updated = np.convolve(capdata.reshape(-1), np.ones(w)*1/w, mode = 'same')

    elif dataset_name == 'NASA PCoE':
      w = 10
      median_trend = medfilt(capdata, kernel_size=5)
      cap_updated = np.convolve(median_trend.reshape(-1), np.ones(w)*1/w, mode = 'same')

    elif dataset_name == 'Unknown':
      window_size = 5  # tune this based on spike width
      cap_updated = maximum_filter1d(capdata, size=window_size)

    else:
      cap_updated = capdata

    return cap_updated

def collate_batch(batch):
  """
  Pad a list of battery samples into batch tensors.

  ``batch`` is the one-element list produced by a ``DataLoader`` whose
  sampler yields index lists (``batch[0]`` is the list returned by
  :meth:`Battery_Dataset.__getitem__`). FECs are zero-padded to the
  longest battery (``L``) and time steps to the longest FEC (``T``).

  Returns:
      Tuple of
      ``(inp, out, cap, eta, index, gamma, cyc_mask, cap_mask, weights, dataset_names)``
      with shapes (B, L, T, d_u), (B, L, T, 1), (B, L, 1), (B, 6),
      (B, L, T), (B, L, 2), (B, L, T), (B, L), (B,) and a list of B strings.
  """

  #inp_, out_, cap_, task_, index_, fec_, mask_ = [], [], [], [], [], [], []
  ncyc_list = np.asarray([b[0].shape[0] for b in batch[0]])
  cyclen_list = np.asarray([b[0].shape[1] for b in batch[0]])

  max_clen = np.max(cyclen_list).astype(int)
  max_cnum = np.max(ncyc_list).astype(int)
  inp_feats = int(batch[0][0][0].shape[-1])
  task_feats = int(batch[0][0][3].shape[-1])
  fec_feats = int(batch[0][0][5].shape[-1])
  #print(max_clen, max_cnum, inp_feats, task_feats)

  inp_batch = torch.empty((len(batch[0]), max_cnum, max_clen, inp_feats))
  out_batch = torch.empty((len(batch[0]), max_cnum, max_clen, 1))
  cap_batch = torch.empty((len(batch[0]), max_cnum, 1))
  task_batch = torch.empty((len(batch[0]), task_feats))
  cyc_mask_batch = torch.empty((len(batch[0]), max_cnum, max_clen))
  cap_mask_batch = torch.zeros((len(batch[0]), max_cnum))
  index_batch = torch.empty((len(batch[0]), max_cnum, max_clen))
  fec_batch = torch.empty((len(batch[0]), max_cnum, fec_feats))
  dat_name_batch = []
  weights_batch = []

  #print(inp_batch.shape)
  #print(type(batch[0][0]))

  for bid, b in enumerate(batch[0]):
    inp, out, cap, task, indx, fec, mask, weight, dname  = b

    pad_seq_len = max_clen - inp.shape[1]
    pad_num_cycs = max_cnum - inp.shape[0]

    inp_batch[bid] = F.pad(inp.permute(2,0,1), (0, pad_seq_len, 0, pad_num_cycs)).permute(1,2,0)
    out_batch[bid] = F.pad(out.permute(2,0,1), (0, pad_seq_len, 0, pad_num_cycs)).permute(1,2,0)
    cap_batch[bid] = F.pad(cap.permute(1,0), (0, pad_num_cycs)).permute(1,0)
    cyc_mask_batch[bid] = F.pad(mask, (0, pad_seq_len, 0, pad_num_cycs))
    cap_mask_batch[bid,:cap.shape[0]] = 1
    index_batch[bid] = F.pad(indx, (0, pad_seq_len, 0, pad_num_cycs))
    fec_batch[bid] = F.pad(fec, (0,0,0, pad_num_cycs))
    task_batch[bid] = torch.from_numpy(task)
    dat_name_batch.append(dname)
    weights_batch.append(weight)

  return (inp_batch.float(), out_batch.float(), cap_batch.float(), task_batch.float(), index_batch.int(), fec_batch.float(), cyc_mask_batch.int(), cap_mask_batch.int(), torch.tensor(weights_batch).float(), dat_name_batch)

class RandomBatchSampler(BatchSampler):
    """
    Shuffled batch sampler over a fixed index range.

    With ``mode='train'`` the indices ``0 .. num_shots-1`` are sampled,
    otherwise ``num_shots .. len(data_ids)-1``. Kept for single-dataset
    (``cap_mode``) experiments; the paper's experiments use
    :class:`Battery_Batch_Sampler`.
    """
    def __init__(self, data_ids, batch_size, num_shots, mode = 'train', drop_last=False):
        #self.data_source = data_source
        self.data_ids = data_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(data_ids)
        self.mode = mode
        self.num_shots = num_shots

    def __iter__(self):
        """Yield batches of indices (see the class docstring)."""
        if self.mode == 'train':
          indices = list(np.arange(0,self.num_shots))
        else:
          indices = list(np.arange(self.num_shots, self.num_samples))

        random.shuffle(indices)
        batches = [
            indices[i:i + self.batch_size]
            for i in range(0, self.num_samples, self.batch_size)
        ]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        return iter(batches)

    def __len__(self):
        """Number of batches."""
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

class Battery_Batch_Sampler(BatchSampler):

    """
    Length-aware batch sampler that groups batteries of similar size.

    Batteries are shuffled, then sorted by FEC count inside pools of
    ``20 * batch_size`` and by average FEC length inside sub-pools of
    ``5 * batch_size``, so that each batch contains batteries with
    similar ``(L, T)`` and padding is minimised.

    Args:
        bat_path: split folder (``<data_root>/Train`` or ``<data_root>/Test``).
        length_path: path to ``<split>_cycle_numel.npy``, holding the total
            number of time steps of every battery in metadata order.
        batch_size: number of batteries per batch.
        indices: optional list of metadata positions to sample from
            (e.g. ``Battery_Dataset.train_ids``). Note that the yielded
            values are positions *within this list*, which is what
            :class:`Battery_Dataset` expects for ``mode='train'/'val'``.
        num_objs_per_batch: unused.
        drop_last: drop the last incomplete batch when computing ``len``.

    The sampler keeps only metadata rows with ``'Cycle count' > 100``
    while :class:`Battery_Dataset` keeps all rows; both index the same
    rows only if no battery has 100 FECs or fewer, which
    ``data_prep/assemble_dataset.py`` guarantees by removing such batteries.
    """
    def __init__(self, bat_path, length_path, batch_size, indices=None, num_objs_per_batch = 10e10, drop_last=False):

        self.bat_path = bat_path
        self.numel_path = length_path
        bats = os.listdir(bat_path)
        self.bat_paths = [os.path.join(self.bat_path,b) for b in bats if os.path.isdir(os.path.join(self.bat_path,b))]
        self.meta_paths = [os.path.join(b, 'metadata.csv') for b in self.bat_paths]

        self.full_metadata = pd.concat([pd.read_csv(m) for m in self.meta_paths])
        self.full_metadata.reset_index(drop = True, inplace = True)
        #self.full_metadata = self.full_metadata.dropna()
        self.full_metadata = self.full_metadata[self.full_metadata['Cycle count']>100]

        self.cyc_lens = self.full_metadata['Cycle count'].values.astype(int)
        self.cycle_numel = np.load(self.numel_path)

        self.batch_size = batch_size

        if indices is not None:
          self.indices = [(i,self.cyc_lens[s], self.cycle_numel[s]//self.cyc_lens[s]) for i, s in enumerate(indices)]
        else:
          self.indices = [(i, s, self.cycle_numel[i]//s) for i, s in enumerate(self.cyc_lens)]       # self.indices[i] = (i, self.chunk_lengths[i])
        self.drop_last = drop_last
        

    def __iter__(self):

        """Yield batches of battery positions grouped by size."""
        random.shuffle(self.indices)
        pooled_indices = []
        for i in range(0, len(self.indices), self.batch_size*20):
            pooled_indices.extend(sorted(self.indices[i:i+self.batch_size*20], key = lambda x:x[1]))

        pooled_indices = [(x[0],x[2]) for x in pooled_indices]

        doubly_pooled_indices = []
        for i in range(0, len(pooled_indices), self.batch_size*5):
            doubly_pooled_indices.extend(sorted(pooled_indices[i:i+self.batch_size*5], key = lambda x:x[1]))

        doubly_pooled_indices = [x[0] for x in doubly_pooled_indices]

        #self.cyc_lens = [self.cyc_lens[x] for x in pooled_indices]

        for i in range(0, len(doubly_pooled_indices), self.batch_size):
            #print('Sampler ',i)
            gc.collect()
            yield doubly_pooled_indices[i:i + self.batch_size]

    def __len__(self):
        """Number of batches."""
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size