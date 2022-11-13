import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os
import numpy as np

from torch_geometric.nn import GCNConv, GATConv
a = torch.tensor([[1.0,2],[-3,4.0]])
b = torch.tensor([[5.0,6],[7.0,8.0]])

print(a/b )# 除法