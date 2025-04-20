#!/usr/bin/env python3
import os
import argparse
import zipfile
import threading
import queue
import time
import shutil

import pandas as pd
import fiona
import geopandas as gpd
from concurrent.futures import ThreadPoolExecutor

def find_zip_files(root):
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith('.zip'):
                yield os.path.join(dp, fn)

def find_gdb_dirs(root):
    for dp, dns, _ in os.walk(root):
        for d in dns:
            if d.lower().endswith('.gdb'):
                yield os.path.join(dp, d)

def append_to_parquet(gdf, out_fp):
    """Create or append GeoDataFrame to a GeoParquet file using PyArrow."""
    engine = 'pyarrow'
    if not os.path.exists(out_fp):
        gdf.to_parquet(out_fp, index=False, engine=engine)
    else:
        existing = gpd.read_parquet(out_fp)
        combined = pd.concat([existing, gdf], ignore_index=True)
        gpd.GeoDataFrame(combined, geometry=existing.geometry.name) \
           .to_parquet(out_fp, index=False, engine=engine)

def process_zip(zip_path, out_q, error_log, extract_root):
    thread = threading.current_thread().name
    zip_name = os.path.basename(zip_path)
    dest = os.path.join(extract_root, os.path.splitext(zip_name)[0])
    os.makedirs(dest, exist_ok=True)

    print(f"[{thread}] Extracting {zip_name} → {dest}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zp:
            zp.extractall(dest)

        gdbs = list(find_gdb_dirs(dest))
        print(f"[{thread}] → {len(gdbs)} .gdb found in {zip_name}")
        for gdb in gdbs:
            print(f"[{thread}]   Processing GDB: {os.path.basename(gdb)}")
            for layer in fiona.listlayers(gdb):
                print(f"[{thread}]     Reading layer '{layer}'")
                gdf = gpd.read_file(gdb, layer=layer)
                if gdf.empty:
                    print(f"[{thread}]     Skipping empty layer")
                    continue
                # tag with source zip and layer name
                gdf['source_file'] = zip_name
                gdf['layer_name']  = layer
                print(f"[{thread}]     Enqueuing {len(gdf)} features")
                out_q.put(gdf)

        print(f"[{thread}] Finished {zip_name}")
    except zipfile.BadZipFile:
        print(f"[{thread}] ERROR: Bad zip {zip_name}")
        with open(error_log, 'a') as ef:
            ef.write(f"{zip_path}\n")
    except Exception as e:
        print(f"[{thread}] ERROR: {zip_name} → {e}")
        with open(error_log, 'a') as ef:
            ef.write(f"{zip_path} ERROR: {e}\n")

def writer_thread(output_file, in_q, sentinel):
    thread = threading.current_thread().name
    print(f"[{thread}] Writer started")
    while True:
        item = in_q.get()
        if item is sentinel:
            in_q.task_done()
            print(f"[{thread}] Sentinel received, exiting")
            break

        n = len(item)
        src = item['source_file'].iat[0]
        lyr = item['layer_name'].iat[0]
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"[{thread}] Attempt {attempt}: Writing {n} records from {src}/{lyr}")
                append_to_parquet(item, output_file)
                print(f"[{thread}] Success: {n} records from {src}/{lyr}")
                break
            except Exception as e:
                print(f"[{thread}] ERROR on attempt {attempt} writing {src}/{lyr}: {e}")
                time.sleep(2)
                print(f"[{thread}] Retrying write for {src}/{lyr}")

        in_q.task_done()

def main(input_dir, output_file):
    error_log = 'errors.txt'
    extract_root = os.path.join(os.path.dirname(output_file), 'extracted')

    # clean up previous runs
    for fn in (output_file, error_log):
        if os.path.exists(fn):
            os.remove(fn)
    if os.path.exists(extract_root):
        shutil.rmtree(extract_root)
    os.makedirs(extract_root, exist_ok=True)

    zips = list(find_zip_files(input_dir))
    print(f"[Main] {len(zips)} zip files found")

    q = queue.Queue()
    sentinel = object()

    wt = threading.Thread(
        target=writer_thread,
        name='Writer',
        args=(output_file, q, sentinel),
        daemon=True
    )
    wt.start()

    with ThreadPoolExecutor(max_workers=5) as executor:
        for zp in zips:
            executor.submit(process_zip, zp, q, error_log, extract_root)

    print("[Main] All zip tasks scheduled")

    q.join()
    print("[Main] All data written")

    q.put(sentinel)
    wt.join()
    print("[Main] Done.")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Multi‑threaded .zip→.gdb→GeoParquet aggregator"
    )
    p.add_argument('input_dir', help="Folder with .zip files")
    p.add_argument('output_file', help="GeoParquet output path")
    args = p.parse_args()
    main(args.input_dir, args.output_file)