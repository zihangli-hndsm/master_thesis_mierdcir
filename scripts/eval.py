import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MTCIRDataset
import chromadb


class ScheiEvaluator:
    def __init__(self, dbpath, lmdb_path, json_path, dataset_cls=MTCIRDataset, **dataset_kwargs):
        self.client = chromadb.PersistentClient(path=dbpath)
        self.json_path = json_path
        self.dataset = dataset_cls(data_path='data', lmdb_path=lmdb_path, json_path=json_path, **dataset_kwargs)
