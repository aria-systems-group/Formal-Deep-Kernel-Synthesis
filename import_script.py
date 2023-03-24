import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import sys
import polytope as pc
from shapely.geometry import Polygon
import itertools
import yaml
import scipy
import scipy.stats as stats
import scipy.optimize as scipy_opt
from scipy.sparse import csr_matrix, lil_matrix, vstack
import os.path
import subprocess
import pickle
import matplotlib.pyplot as plt
import progressbar
# from alive_progress import alive_bar
import copy
import os
import shutil
import gpytorch

def error_print(str):
    print(str)
    exit()