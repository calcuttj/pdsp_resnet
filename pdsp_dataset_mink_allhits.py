from torch.utils.data import Dataset, DataLoader

import MinkowskiEngine as ME

import torch
import torch.nn as nn

#import process_all_hits
from dataclasses import dataclass

@dataclass
class Config:
  batch_size: int
  num_workers: int = 1


##Only uses plane 2
class PDSPDataset(Dataset):
  def __init__(self, pdsp_data):
    super().__init__()
    self.pdsp_data = pdsp_data

  def __len__(self):
    return self.pdsp_data.nevents

  def __getitem__(self, i: int) -> dict:
    (locs, features), label = self.pdsp_data.get_plane(i, 2)
    #label = self.pdsp_data.topos[i]

    return {
      'coordinates': torch.from_numpy(locs).to(torch.long),
      'features': torch.from_numpy(features).to(torch.float32),
      'label': torch.from_numpy(label).to(torch.float32)
    }

def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }

def get_loader(pdsp_data, config):
  dataset = PDSPDataset(pdsp_data)

  return DataLoader(
    dataset,
    num_workers=1, ##config.num_workers
    collate_fn=minkowski_collate_fn,
    batch_size=config.batch_size,
  )
