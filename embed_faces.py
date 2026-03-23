#!/usr/bin/env python3
import os
import json
import logging
import argparse
import subprocess
import tempfile
from datetime import datetime

def setup_logger(debug_mode):
    """Sets up separate logging for console, main log file, and a dedicated debug log file in a 'logs' subfolder."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_filename = os.path.join(log_dir, f"face_embed_log_{timestamp}.txt")
    debug_log_filename = None
    
    logger = logging.getLogger("FaceEmbedder")
    # Base logger must be DEBUG if debug_mode is on, so debug messages aren't dropped entirely
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Main File Handler (Always INFO and above)
    fh_main = logging.FileHandler(main_log_filename, encoding='utf-8')
    fh_main.setLevel(logging.INFO)
    fh_main.setFormatter(formatter)
    logger.addHandler(fh_main)
    
    # 2. Console Handler (Always INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s')) # Cleaner format for console
    logger.addHandler(ch)
    
    # 3. Dedicated Debug File Handler (Only active if --debug flag is passed)
    if debug_mode:
        debug_log_filename = os.path.join(log_dir, f"face_embed_debug_{timestamp}.txt")
        fh_debug = logging.FileHandler(debug_log_filename, encoding='utf-8')
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        logger.addHandler(fh_debug)
    
    return logger, main_log_filename, debug_log_filename

def find_image_xmp_pairs(root_dir):
    """Recursively finds images that have an exact matching .xmp sidecar."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    pairs =[]
    
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                img_path = os.path.join(dirpath, file)
                xmp_path = img_path + '.xmp'
                
                if os.path.exists(xmp_path):
                    pairs.append((os.path.abspath(img_path), os.path.abspath(xmp_path)))
                    
    return pairs

def is_duplicate_region(reg1, reg2):
    """Checks if two face regions are effectively identical (Name and Coordinates)."""
    if reg1.get('Name') != reg2.get('Name'): return False
    
    area1 = reg1.get('Area', {})
    area2 = reg2.get('Area', {})
    
    for k in['X', 'Y', 'W', 'H']:
        if abs(float(area1.get(k, 0)) - float(area2.get(k, 0))) > 0.001: 
            return False
    return True

def normalize_list(data):
    """Ensures Exiftool list outputs are flat lists."""
    if data is None: return[]
    if isinstance(data, str): return[data]
    return list(data)

def encode_exiftool_struct(val):
    """
    Recursively encodes Python dicts/lists into ExifTool's native structure serialization format.
    """
    if isinstance(val, dict):
        parts =[]
        for k, v in val.items():
            parts.append(f"{k}={encode_exiftool_struct(v)}")
        return "{" + ",".join(parts) + "}"
    elif isinstance(val, list):
        parts =[encode_exiftool_struct(item) for item in val]
        return "[" + ",".join(parts) + "]"
    else:
        s = str(val)
        s = s.replace('|', '||').replace(',', '|,').replace('}', '|}').replace(']', '|]')
        if s and s[0] in '{[ \t\r\n':
            s = '|' + s
        return s

def main():
    parser = argparse.ArgumentParser(description="Embed Face metadata from XMP sidecars to Images using ExifTool.")
    parser.add_argument('directory', help="Target root directory to scan.")
    parser.add_argument('--merge', action='store_true', help="Merge new XMP tags with existing image tags instead of skipping.")
    parser.add_argument('--keep-original', action='store_true', help="Keep original backup files (do not use -overwrite_original).")
    parser.add_argument('--include-unknown', action='store_true', help="Include face regions and subjects named 'Unknown' (they are ignored by default).")
    parser.add_argument('--debug', action='store_true', help="Save verbose debugging output to a separate log file.")
    args = parser.parse_args()

    logger, log_file, debug_file = setup_logger(args.debug)
    logger.info(f"Starting Face Embedder run. Directory: {args.directory}")
    
    if args.debug:
        logger.info(f"Debug mode enabled. Detailed debug info will be saved to: {debug_file}")
    
    # 1. Find Files
    pairs = find_image_xmp_pairs(args.directory)
    logger.info(f"Found {len(pairs)} image/xmp pairs to evaluate.")
    
    if not pairs:
        logger.info("No matching pairs found. Exiting.")
        return

    # 2. PASS 1: Read all metadata efficiently
    logger.info("Pass 1: Reading existing metadata from files (this may take a moment)...")
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        for img, xmp in pairs:
            f.write(f"{img}\n{xmp}\n")
        read_args_file = f.name

    try:
        cmd_read =['exiftool', '-charset', 'utf8', '-j', '-struct', '-Subject', '-RegionInfo', '-@', read_args_file]
        logger.debug(f"Executing Read Command: {' '.join(cmd_read)}")
        
        result = subprocess.run(cmd_read, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0 and not result.stdout:
            logger.error(f"ExifTool Read Error: {result.stderr}")
            return
            
        raw_metadata = json.loads(result.stdout)
        meta_map = {os.path.abspath(item['SourceFile']): item for item in raw_metadata}
        logger.debug(f"--- RAW READ METADATA ---\n{json.dumps(raw_metadata, indent=2)}\n-------------------------")
        
    except Exception as e:
        logger.error(f"Failed to parse Pass 1 output: {e}")
        return
    finally:
        os.remove(read_args_file)

    # 3. Evaluate Data
    write_tasks =[]
    
    for img_path, xmp_path in pairs:
        img_meta = meta_map.get(img_path, {})
        xmp_meta = meta_map.get(xmp_path, {})
        
        # Parse XMP Data
        xmp_subjects = normalize_list(xmp_meta.get('Subject'))
        xmp_region_info = xmp_meta.get('RegionInfo', {})
        xmp_regions = normalize_list(xmp_region_info.get('RegionList') if isinstance(xmp_region_info, dict) else[])
        
        # Filter out "Unknown" names if flag is not set
        if not args.include_unknown:
            xmp_subjects =[s for s in xmp_subjects if str(s).strip().lower() != 'unknown']
            
            filtered_regions =[]
            for r in xmp_regions:
                r_name = str(r.get('Name', '')).strip().lower()
                if r_name != 'unknown':
                    filtered_regions.append(r)
            xmp_regions = filtered_regions

        # Parse existing Image Data
        img_subjects = normalize_list(img_meta.get('Subject'))
        img_region_info = img_meta.get('RegionInfo', {})
        img_regions = normalize_list(img_region_info.get('RegionList') if isinstance(img_region_info, dict) else[])
        
        # Check if there is anything left to copy after filtering
        if not xmp_subjects and not xmp_regions:
            logger.warning(f"Skipped {img_path}: XMP sidecar contains no valid Subject or RegionInfo (or they were ignored as 'Unknown').")
            continue
            
        # Check if Image already has data and evaluate skip vs merge
        if (img_subjects or img_regions) and not args.merge:
            logger.warning(f"Skipped {img_path}: Image already contains metadata (Subject or Regions) and --merge flag is off.")
            continue

        # -- Auto Populate Subjects --
        for region in xmp_regions:
            if region.get('Type') == 'Face' and region.get('Name'):
                name = region.get('Name')
                if name not in xmp_subjects:
                    xmp_subjects.append(name)

        # -- Prepare Merged Subjects --
        final_subjects = list(img_subjects) if args.merge else[]
        for subj in xmp_subjects:
            if subj not in final_subjects:
                final_subjects.append(subj)

        # -- Prepare Merged Regions --
        final_regions = list(img_regions) if args.merge else[]
        for x_reg in xmp_regions:
            is_dup = False
            for f_reg in final_regions:
                if is_duplicate_region(x_reg, f_reg):
                    is_dup = True
                    break
            if not is_dup:
                final_regions.append(x_reg)
                
        # Reconstruct RegionInfo Dictionary
        final_region_info = {}
        dims = xmp_region_info.get('AppliedToDimensions')
        if not dims and isinstance(img_region_info, dict):
            dims = img_region_info.get('AppliedToDimensions')
            
        if dims:
            final_region_info['AppliedToDimensions'] = dims
        if final_regions:
            final_region_info['RegionList'] = final_regions

        logger.debug(f"--- FINAL PREPARED REGION DICTIONARY FOR {os.path.basename(img_path)} ---\n{json.dumps(final_region_info, indent=2)}\n---------------------------------------------------------")
        write_tasks.append((img_path, final_subjects, final_region_info))

    if not write_tasks:
        logger.info("No files require modification after evaluation.")
        return

    # 4. PASS 2: Write metadata
    logger.info(f"Pass 2: Applying updates to {len(write_tasks)} files...")
    
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
        for img_path, subjects, region_info in write_tasks:
            
            # Clear existing lists to prevent duplicates when appending
            f.write("-Subject=\n") 
            for subj in subjects:
                f.write(f"-Subject={subj}\n")
                
            # Write RegionInfo structure
            if region_info:
                exiftool_struct_str = encode_exiftool_struct(region_info)
                logger.debug(f"--- EXIFTOOL STRUCT STRING --- \n{exiftool_struct_str}\n------------------------------")
                f.write(f"-RegionInfo={exiftool_struct_str}\n")
            elif not args.merge:
                f.write("-RegionInfo=\n")
                
            if not args.keep_original:
                f.write("-overwrite_original\n")
                
            f.write(f"{img_path}\n")
            f.write("-execute\n")
            
        write_args_file = f.name
        
    if args.debug:
        logger.debug(f"--- CONTENTS OF WRITE ARGUMENT FILE ---")
        with open(write_args_file, 'r', encoding='utf-8') as dbg_file:
            logger.debug("\n" + dbg_file.read() + "---------------------------------------")

    try:
        cmd_write =['exiftool', '-charset', 'utf8', '-@', write_args_file]
        logger.debug(f"Executing Write Command: {' '.join(cmd_write)}")
        
        result = subprocess.run(cmd_write, capture_output=True, text=True, encoding='utf-8')
        
        logger.debug(f"ExifTool Write STDOUT:\n{result.stdout}")
        logger.debug(f"ExifTool Write STDERR:\n{result.stderr}")
        
        for task in write_tasks:
            logger.info(f"Processed / Updated: {task[0]}")
            
        # Construct final success message based on whether debug logging is enabled
        success_msg = f"Success! {len(write_tasks)} files updated. Check {log_file} for details."
        logger.info(success_msg)
        
    except Exception as e:
        logger.error(f"Pass 2 failed while writing metadata: {e}")
    finally:
        os.remove(write_args_file)

if __name__ == "__main__":
    main()