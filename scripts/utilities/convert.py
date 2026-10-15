import os
import glob
import numpy as np
import tifffile
import shutil
from tqdm import tqdm

marida_map = {
    0: 0,
    1: 2,
    2: 3,
    3: 3,
    4: 4,
    5: 4,
    6: 0,
    7: 1,
    8: 1,
    9: 4,
    10: 1,
    11: 1,
    12: 1,
    13: 0,
    14: 1,
    15: 1
}

mados_map = {
    0: 0,
    1: 2,
    2: 3,
    3: 3,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    8: 1,
    9: 4,
    10: 1,
    11: 1,
    12: 1,
    13: 4,
    14: 4,
    15: 4
}

def remap_mask(mask_path, new_mask_path, mapping_dict):
    img = tifffile.imread(mask_path).astype(int)
    
    # We create a palette array that maps old_index to new_index
    max_val = max(mapping_dict.keys())
    palette = np.zeros(max_val + 1, dtype=np.uint8)
    for k, v in mapping_dict.items():
        palette[k] = v
        
    # Map all values
    new_img = palette[img]
    
    os.makedirs(os.path.dirname(new_mask_path), exist_ok=True)
    # Write back without breaking standard tiff format
    tifffile.imwrite(new_mask_path, new_img, compression='zlib')

def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def process_marida(src_dir, dst_dir):
    print("Processing MARIDA...")
    files = glob.glob(os.path.join(src_dir, "patches", "**", "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    for f in tqdm(files, desc="MARIDA files"):
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        if f.endswith('_cl.tif'):
            remap_mask(f, dst_path, marida_map)
        else:
            copy_file(f, dst_path)

def process_mados(src_dir, dst_dir):
    print("Processing MADOS...")
    files = glob.glob(os.path.join(src_dir, "**", "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    for f in tqdm(files, desc="MADOS files"):
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        # In MADOS, class masks are usually named like Scene_0_L2R_cl_1.tif
        if '_cl_' in os.path.basename(f) and f.endswith('.tif'):
            remap_mask(f, dst_path, mados_map)
        else:
            copy_file(f, dst_path)

if __name__ == "__main__":
    # Resolve paths relative to the script's location (assuming it's in scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "data")
    
    process_marida(os.path.join(base_dir, "MARIDA"), os.path.join(base_dir, "MARIDA_5Class"))
    process_mados(os.path.join(base_dir, "MADOS"), os.path.join(base_dir, "MADOS_5Class"))
    print("Conversion finished successfully!")
