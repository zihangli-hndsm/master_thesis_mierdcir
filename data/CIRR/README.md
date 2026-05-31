# CIRR Local Data

Place CIRR files in this directory before running evaluation or LMDB creation.

Expected local files:

```text
data/CIRR/
├── cap.rc2.val.json
├── cap.rc2.test1.json
├── split.rc2.val.json
├── split.rc2.test1.json
├── img_raw_filtered/
└── images_224_lmdb/
```

Build the LMDB after restoring raw images:

```bash
cd data/CIRR
python save_lmdb.py
```

The raw images, JSON split files, archives, and generated LMDB are ignored by git.

See the repository-level `DATASET_NOTICE.md` before using or sharing any CIRR-related local artifacts.
