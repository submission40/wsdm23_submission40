import numpy as np
import pandas as pd
import tqdm
from data_manager import DataManager

import math
def prepare_knn_data(data_manager):
# prepare input for knn models
  data_manager.load_playlist_track()
  pt = data_manager.playlist_track
  n_int = len(pt.data)
  df_data = np.zeros((n_int, 3), dtype=np.int64)
  for pid in tqdm.tqdm(range(data_manager.n_playlists)):
    df_data[pt.indptr[pid]:pt.indptr[pid+1],0] = pid
    df_data[pt.indptr[pid]:pt.indptr[pid+1], 1] = pt.indices[pt.indptr[pid]:pt.indptr[pid+1]]
    df_data[pt.indptr[pid]:pt.indptr[pid+1], 2] = pt.data[pt.indptr[pid]:pt.indptr[pid+1]]

  cols = ["SessionId", "ItemId", "Pos"]
  df_data = pd.DataFrame(df_data, columns=cols)
  playlist_data = pd.read_hdf("knn_data/playlist_data")
  df_data = df_data.merge(playlist_data[["pid", "modified_at"]].rename(columns={"pid":"SessionId"}), on="SessionId")
  df_data = df_data.merge(pd.DataFrame(data_manager.tracks_info.values())[["id", "duration_ms"]].rename(columns={'id':"ItemId"}), on="ItemId")
  df_data.sort_values(["SessionId", "Pos"], ascending = [True,  False], inplace=True)
  df_data["duration_sum"] = df_data.groupby('SessionId')["duration_ms"].cumsum()
  df_data["Time"] = df_data["modified_at"] - df_data["duration_sum"]
  return df_data

if __name__ == "__main__":
    out_path = sys.argv[1]
data_manager = DataManager()
df_data = prepare_knn_data(data_manager)
df_data.to_hdf("knn_data/df_data", "abc")