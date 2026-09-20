import lmdb
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# Reproduction use:
# 1. Place the MTCIR image folders under `data/MTCIR/images/`. The expected
#    structure is numeric subfolders, for example `images/001/001234.jpg`.
# 2. Run from the MTCIR folder:
#       cd data/MTCIR
#       python save_lmdb.py
# 3. The output `images_224_lmdb/` is used by `eval_checkpoint.py` for MTCIR
#    and MerdCIR. Keys are stored as `subdir/file_name`, for example
#    `001/001234.jpg`, so JSONL image ids must use the same relative format.
# 4. Do not commit the generated LMDB. It is reproducible local data and is
#    ignored by the repository `.gitignore`.


def create_lmdb(image_root, output_path):
    # Estimate LMDB map size for roughly 500k resized 224px images.
    # Increase this if LMDB raises MDB_MAP_FULL for a larger local dataset.
    map_size = 50 * 1024 * 1024 * 1024 
    env = lmdb.open(output_path, map_size=map_size)

    with env.begin(write=True) as txn:
        # Traverse numeric image subdirectories.
        for subdir in tqdm(os.listdir(image_root)):
            subdir_path = os.path.join(image_root, subdir)
            if not os.path.isdir(subdir_path): continue
            
            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                # Store keys as "folder/file", for example "001/123.jpg".
                # This key format must match the `reference_path` and
                # `target_path` values in the MTCIR/MerdCIR JSONL files.
                key = f"{subdir}/{img_name}".encode('ascii')
                
                # Resize to 224px and serialize before writing.
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB').resize((224, 224))
                        buffer = BytesIO()
                        img.save(buffer, format="JPEG", quality=90)
                        val = buffer.getvalue()
                        txn.put(key, val)
                except:
                    continue
    env.close()


if __name__ == "__main__":
    # Run once after downloading or restoring the raw MTCIR images.
    # Change these defaults if your image folder or LMDB destination differs.
    create_lmdb("./images", "./images_224_lmdb")
