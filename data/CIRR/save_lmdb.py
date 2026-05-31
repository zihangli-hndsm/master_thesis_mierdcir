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
    将文件夹下的所有图片递归写入 LMDB
    :param image_root: 原始图片根目录
    :param output_path: LMDB 数据库保存路径
    :param map_size_gb: 数据库最大容量（单位 GB）
    """
    
    # 1. 统计总图片数量（用于显示进度条）
    print("正在扫描文件...")
    all_files = []
    for root, dirs, files in os.walk(image_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_files.append(os.path.join(root, f))
    
    total_count = len(all_files)
    print(f"找到图片总数: {total_count}")

    # 2. 初始化 LMDB 环境
    # map_size 定义了数据库的最大容量
    map_size = map_size_gb * 1024 * 1024 * 1024 
    env = lmdb.open(output_path, map_size=map_size)

    # 3. 开始写入
    print("正在开始写入 LMDB...")
    with env.begin(write=True) as txn:
        for img_path in tqdm(all_files, desc="Processing Images"):
            # 生成 key：例如 "一级/二级/三级/文件名.jpg"
            # 使用 relpath 获取相对路径，replace 确保在 Windows 上也是 "/" 分隔符
            # The downstream dataset loader queries this exact relative key.
            # If retrieval later reports missing images, first confirm that the
            # JSON image ids match these LMDB keys.
            rel_path = os.path.relpath(img_path, image_root)
            key = rel_path.replace("\\", "/").encode('ascii')
            
            try:
                with Image.open(img_path) as img:
                    # 统一转为 RGB 模式（防止 RGBA 或灰度图报错）
                    img = img.convert('RGB')
                    
                    # 缩放到指定尺寸
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    # 序列化为二进制流
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=90)
                    val = buffer.getvalue()
                    
                    # 写入数据库
                    txn.put(key, val)
            except Exception as e:
                print(f"\n跳过损坏图片 {img_path}: {e}")
                continue

    # 关闭环境
    env.close()
    print(f"\n写入完成！LMDB 已保存至: {output_path}")

# --- 运行示例 ---
if __name__ == "__main__":
    # Default paths assume this script is launched from `data/CIRR`.
    # Adjust these two variables if your raw images or output database live
    # somewhere else.
    input_dir = "./img_raw_filtered"
    output_db = "./images_224_lmdb"
    
    create_lmdb(input_dir, output_db, map_size_gb=5)
