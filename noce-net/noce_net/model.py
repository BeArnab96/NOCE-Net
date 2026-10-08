"""
noce_net.model
==============
Top-level NOCE-Net module.
"""
import torch
import torch.nn as nn

from noce_net.layers import CycleBlock, Predictor


class BatteryModel(nn.Module):
  """
  Container for NOCE-Net.

  Attributes:
      cycle_model: :class:`~noce_net.layers.CycleBlock`, the voltage-response
          model that is trained and evaluated in the paper.
      cap_model: :class:`~noce_net.layers.Predictor`, an auxiliary capacity
          head. It is instantiated so that checkpoints keep the same
          ``state_dict`` keys, but it is not trained or used by any script in
          this repository.

  Args:
      configs: ``model_configs`` dictionary (see :func:`noce_net.config.load_config`).
  """
  def __init__(self, configs):
    super().__init__()
    self.configs = configs
    self.cycle_model = CycleBlock(configs)
    self.cap_model= Predictor(configs)
