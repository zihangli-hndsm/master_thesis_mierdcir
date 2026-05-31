# MTCIR Local Data

Place MTCIR files in this directory before training or evaluation.

Training images come from the LLaVA-pretrain dataset available on Hugging Face, which is a subset of Google's CC3M.

Expected local files:

```text
data/MTCIR/
├── mtcir.jsonl
├── images/
│   └── <subfolder>/<image>.jpg
└── images_224_lmdb/
```

Build the LMDB:

```bash
cd data/MTCIR
python save_lmdb.py
```

The LMDB keys are stored as `subfolder/image_name`, so JSONL image paths must use the same relative format.

Raw images, JSONL data, and generated LMDBs are ignored by git.

These images may contain harmful or sensitive content. See the repository-level `DATASET_NOTICE.md` before using or sharing MTCIR-related local artifacts.
