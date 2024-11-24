import os
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Dataset options
#
__C.TRAINLOG = edict()
__C.TRAINLOG.DATA_NAMES = []
__C.TRAINLOG.LOGJSON=None
__C.TRAINLOG.LOGTXT=None
__C.TRAINLOG.STARTTIME=None
__C.TRAINLOG.TR_FIG_METRIC=None
__C.TRAINLOG.VAL_FIG_METRIC=None
__C.TRAINLOG.ITER_PATH=None
__C.TRAINLOG.NETWORK_DICT = {}
__C.TRAINLOG.EXCEL_LOG=None
__C.TRAINLOG.EXCEL_LOGDetail=None

__C.TRAINLOG.EXCEL_LOGSheet=[]
__C.TRAINLOG.EXCEL_LOGSheetDetail=[]
__C.TRAINLOG.EXCEL_LOGDetailAvg=[]



__C.DA = edict()
__C.DA.BN_DOMAIN_MAP={}
__C.DA.S=0
__C.DA.T=1
__C.DA.NUM_DOMAINS_BN=1