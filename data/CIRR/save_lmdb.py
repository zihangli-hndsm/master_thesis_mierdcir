import lmdb
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# Reproduction use:
# 1. Put the raw CIRR images under `data/CIRR/img_raw_filtered/` or update
#    `input_dir` in the main block below.
# 2. Run from the CIRR folder:
#       cd data/CIRR
#       python save_lmdb.py
# 3. The script writes `images_224_lmdb/`, which is the path expected by
#    `eval_checkpoint.py` for CIRR. The LMDB keys are image paths relative to
#    `image_root`, using "/" separators.
# 4. Keep the output LMDB out of git. It is generated data and is ignored by
#    the repository `.gitignore`.


def create_lmdb(image_root, output_path, map_size_gb=50):
    """
    Recursively write all images under the folder to LMDB
    :param image_root: source image root directory
    :param output_path: LMDB output path
    :param map_size_gb: maximum database size in GB
    """
    
    # 1. Count images first so the progress bar has a stable total.
    print("Scanning files...")
    all_files = []
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_files.append(os.path.join(root, f))
    
    total_count = len(all_files)
    print(f"Total images found: {total_count}")

    # 2. Initialize the LMDB environment.
    # Reproducibility note: keep this step explicit for repeatable experiments.
    map_size = map_size_gb * 1024 * 1024 * 1024 
    env = lmdb.open(output_path, map_size=map_size)

    # 3. Write image records into LMDB.
    print("Writing image records into LMDB...")
    with env.begin(write=True) as txn:
        for img_path in tqdm(all_files, desc="Processing Images"):
            # Create an LMDB key from the relative image path.
            # Normalize relative paths to forward slashes across platforms.
            # The downstream dataset loader queries this exact relative key.
            # If retrieval later reports missing images, first confirm that the
            # JSON image ids match these LMDB keys.
            rel_path = os.path.relpath(img_path, image_root)
            key = rel_path.replace("\\", "/").encode('ascii')
            
            try:
                with Image.open(img_path) as img:
                    # Convert images to RGB to avoid RGBA/grayscale shape mismatches.
                    img = img.convert('RGB')
                    
                    # Resize images to the configured size.
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # Serialize the processed image to bytes.
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=90)
                    val = buffer.getvalue()
                    
                    # Write the image bytes to LMDB.
                    txn.put(key, val)
            except Exception as e:
                print(f"\nSkipping corrupt image {img_path}: {e}")
                continue

    # Close the LMDB environment.
    env.close()
    print(f"\nWrite complete; LMDB saved to: {output_path}")

# --- Example invocation. ---
if __name__ == "__main__":
    # Default paths assume this script is launched from `data/CIRR`.
    # Adjust these two variables if your raw images or output database live
    # somewhere else.
    input_dir = "./img_raw_filtered"
    output_db = "./images_224_lmdb"
    
    create_lmdb(input_dir, output_db, map_size_gb=5)
