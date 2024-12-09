import pandas as pd

data = pd.read_csv('data_small_multiTW.txt', sep=r'\s+', header=None,
                       names=["LOC_ID", "XCOORD", "YCOORD", "DEMAND", "SERVICETIME", "NUM_TW",
                              "READYTIME1", "DUETIME1", "READYTIME2", "DUETIME2", "READYTIME3", "DUETIME3"])