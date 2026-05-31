# FashionIQ Local Data

Place FashionIQ files in this directory before running FashionIQ evaluation.

Expected local files:

```text
data/FashionIQ/
├── captions/
│   └── cap.{dress,shirt,toptee}.{val,test}.json
├── image_splits/
│   └── split.{dress,shirt,toptee}.{val,test}.json
├── images/
│   └── asin2url.{dress,shirt,toptee}.txt
├── images_val_224_lmdb/
└── images_test_224_lmdb/
```

Build LMDBs:

```bash
cd data/FashionIQ
python save_lmdb.py --data-root . --split val
python save_lmdb.py --data-root . --split test
```

The downloaded images, URL maps, split JSON files, and generated LMDBs are ignored by git.

See the repository-level `DATASET_NOTICE.md` before using or sharing any FashionIQ-related local artifacts.
