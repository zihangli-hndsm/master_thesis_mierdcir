# LaSCo Local Data

Place LaSCo/COCO-derived files in this directory before training or evaluation.

Expected local files:

```text
data/LaSCo/
├── captions/
│   ├── lasco_train.json
│   └── lasco_val.json
├── images/
│   ├── train2014/
│   └── val2014/
└── images_224_lmdb/
```

Build the LMDB:

```bash
python data/LaSCo/save_lmdb.py \
  --image-root data/LaSCo/images \
  --output-lmdb data/LaSCo/images_224_lmdb
```

Downloaded images, captions, archives, and generated LMDBs are ignored by git.

See the repository-level `DATASET_NOTICE.md` before using or sharing any LaSCo-related local artifacts.
