import os
import glob
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False
    print("[prepare_cicids] WARNING: pyarrow not available; will write gzipped CSVs instead of parquet.")

def clean_column_names(cols):
    # strip whitespace and normalize
    new = [c.strip() for c in cols]
    return new

def sane_numeric_fix(df_chunk):
    df_chunk = df_chunk.copy()
    df_chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    numcols = df_chunk.select_dtypes(include=[np.number]).columns
    medians = df_chunk[numcols].median()

    # If an entire column is NaN, median will be NaN → replace that case with 0
    medians = medians.fillna(0)

    df_chunk[numcols] = df_chunk[numcols].fillna(medians)

    return df_chunk


def process_single_csv(infile, out_dir, chunksize=200_000, drop_cols=None, label_col="Label", multiclass=True):
    if drop_cols is None:
        drop_cols = []  # your dataset doesn't have Flow ID / IP columns

    base = os.path.splitext(os.path.basename(infile))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + (".parquet" if _HAS_PYARROW else ".csv.gz"))

    writer = None
    first_chunk = True
    total_rows = 0
    class_counts = {}

    for chunk in pd.read_csv(infile, chunksize=chunksize, iterator=True):
        # normalize column names
        chunk.columns = clean_column_names(list(chunk.columns))

        # detect duplicate column names; drop the second 'Fwd Header Length' if present
        cols = list(chunk.columns)
        if 'Fwd Header Length' in set([c for c in cols if cols.count(c) > 1]):
            # keep only first occurrence of duplicated columns with same name
            seen = set()
            keep_cols = []
            for c in cols:
                if c in seen:
                    continue
                keep_cols.append(c)
                seen.add(c)
            chunk = chunk[keep_cols]
            cols = list(chunk.columns)

        # drop user-specified id cols if present
        chunk.drop(columns=[c for c in (drop_cols or []) if c in chunk.columns], inplace=True, errors='ignore')

        # Clean numeric columns
        chunk = sane_numeric_fix(chunk)

        # For multiclass: keep original Label column as-is and accumulate counts
        if multiclass and label_col in chunk.columns:
            vals = chunk[label_col].astype(str).value_counts().to_dict()
            for k, v in vals.items():
                class_counts[k] = class_counts.get(k, 0) + int(v)
        elif not multiclass:
            # binary mapping (not used here)
            if label_col in chunk.columns:
                chunk['target'] = (chunk[label_col].astype(str).str.upper() != 'BENIGN').astype(int)
                vals = chunk['target'].value_counts().to_dict()
                for k, v in vals.items():
                    class_counts[str(k)] = class_counts.get(str(k), 0) + int(v)

        # write chunk to parquet/csv
        if _HAS_PYARROW:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if first_chunk:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)
                first_chunk = False
            else:
                writer.write_table(table)
        else:
            if first_chunk:
                chunk.to_csv(out_path, index=False, compression='gzip', mode='w')
                first_chunk = False
            else:
                chunk.to_csv(out_path, index=False, compression='gzip', mode='a', header=False)

        total_rows += len(chunk)

    if _HAS_PYARROW and writer is not None:
        writer.close()

    meta = {
        'input_file': infile,
        'output_file': out_path,
        'rows': total_rows,
        'class_counts': class_counts
    }
    print(f"[prepare_cicids] processed {infile} -> {out_path} ({total_rows} rows)")
    return meta

def merge_parquets(input_folder, merged_outpath, sample_frac=None, stratified=False, label_col='Label', random_state=42):
    os.makedirs(os.path.dirname(merged_outpath), exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_folder, "*.parquet")))
    if len(files) == 0:
        raise FileNotFoundError("No parquet files found in " + input_folder)

    if sample_frac is None or sample_frac >= 1.0:
        # simple merge
        first = True
        writer = None
        for f in files:
            tbl = pq.read_table(f)
            df = tbl.to_pandas()
            if first:
                writer = pq.ParquetWriter(merged_outpath, pa.Table.from_pandas(df, preserve_index=False).schema, compression='snappy')
                writer.write_table(pa.Table.from_pandas(df, preserve_index=False))
                first = False
            else:
                writer.write_table(pa.Table.from_pandas(df, preserve_index=False))
        if writer:
            writer.close()
        print(f"[prepare_cicids] merged {len(files)} files -> {merged_outpath}")
        return merged_outpath

    # sampling path (supports stratified sampling by label_col)
    if stratified:
        print("[prepare_cicids] performing stratified sampling")
        # compute total class counts
        total_counts = {}
        for f in files:
            tbl = pq.read_table(f, columns=[label_col])
            dfc = tbl.to_pandas()
            vc = dfc[label_col].astype(str).value_counts().to_dict()
            for k, v in vc.items():
                total_counts[k] = total_counts.get(k, 0) + int(v)
        min_samples = 20
        desired = {k: max(min_samples, int(v * sample_frac)) for k, v in total_counts.items()}
        print('[prepare_cicids] desired per-class sample (preview):', dict(list(desired.items())[:10]))

        first = True
        writer = None
        running = {k: 0 for k in desired.keys()}
        for f in files:
            df = pq.read_table(f).to_pandas()
            out_chunks = []
            for cls, want in desired.items():
                need = want - running.get(cls, 0)
                if need <= 0:
                    continue
                rows = df[df[label_col].astype(str) == cls]
                if rows.shape[0] == 0:
                    continue
                take = min(need, rows.shape[0])
                sampled = rows.sample(n=take, random_state=random_state)
                out_chunks.append(sampled)
                running[cls] = running.get(cls, 0) + take
            if len(out_chunks) == 0:
                continue
            towrite = pd.concat(out_chunks, ignore_index=True)
            if first:
                writer = pq.ParquetWriter(merged_outpath, pa.Table.from_pandas(towrite, preserve_index=False).schema, compression='snappy')
                writer.write_table(pa.Table.from_pandas(towrite, preserve_index=False))
                first = False
            else:
                writer.write_table(pa.Table.from_pandas(towrite, preserve_index=False))
        if writer:
            writer.close()
        print(f"[prepare_cicids] stratified merged sample written -> {merged_outpath}")
        return merged_outpath

    # non-stratified sample
    first = True
    writer = None
    for f in files:
        df = pq.read_table(f).to_pandas()
        sampled = df.sample(frac=sample_frac, random_state=random_state)
        if first:
            writer = pq.ParquetWriter(merged_outpath, pa.Table.from_pandas(sampled, preserve_index=False).schema, compression='snappy')
            writer.write_table(pa.Table.from_pandas(sampled, preserve_index=False))
            first = False
        else:
            writer.write_table(pa.Table.from_pandas(sampled, preserve_index=False))
    if writer:
        writer.close()
    print(f"[prepare_cicids] non-stratified merged sample written -> {merged_outpath}")
    return merged_outpath

def build_label_encoder(processed_folder, label_col='Label', save_path=None):
    files = sorted(glob.glob(os.path.join(processed_folder, "*.parquet")))
    labels = []
    for f in files:
        tbl = pq.read_table(f, columns=[label_col])
        dfc = tbl.to_pandas()
        labels.extend(dfc[label_col].astype(str).values.tolist())
    le = LabelEncoder()
    le.fit(labels)
    if save_path:
        joblib.dump(le, save_path)
    return le

def main(args):
    csvs = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if len(csvs) == 0:
        raise FileNotFoundError("No CSVs found in " + args.input_dir)
    processed = []
    manifest = []
    for f in csvs:
        meta = process_single_csv(f, out_dir=args.out_dir, chunksize=args.chunksize, drop_cols=args.drop_cols, label_col=args.label_col, multiclass=args.multiclass)
        processed.append(meta['output_file'])
        manifest.append(meta)

    # save manifest
    os.makedirs(args.out_dir, exist_ok=True)
    import json
    manifest_path = os.path.join(args.out_dir, 'processed_manifest.json')
    with open(manifest_path, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print('[prepare_cicids] manifest saved to', manifest_path)

    # if multiclass, optionally build label encoder
    if args.multiclass and args.save_label_encoder:
        if not _HAS_PYARROW:
            raise RuntimeError('pyarrow required to build label encoder from parquet outputs')
        le = build_label_encoder(args.out_dir, label_col=args.label_col, save_path=args.save_label_encoder)
        print('[prepare_cicids] saved LabelEncoder to', args.save_label_encoder)

    # optional merge + sampling
    if args.merge_out:
        if not _HAS_PYARROW:
            raise RuntimeError('pyarrow required for merging parquet files')
        merge_parquets(args.out_dir, args.merge_out, sample_frac=args.sample_frac, stratified=args.stratified, label_col=(args.label_col))
    print('[prepare_cicids] done.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True, help='Folder with raw CSV files')
    p.add_argument('--out_dir', required=True, help='Folder to save processed parquet files (one per CSV)')
    p.add_argument('--chunksize', type=int, default=200_000, help='CSV read chunk size')
    p.add_argument('--drop_cols', nargs='*', default=[], help='Identity columns to drop (not present in your files)')
    p.add_argument('--label_col', default='Label', help='Label column name in raw CSVs')
    p.add_argument('--multiclass', action='store_true', help='If set, keep multiclass labels')
    p.add_argument('--save_label_encoder', default=None, help='Path to save LabelEncoder joblib (use with --multiclass)')
    p.add_argument('--merge_out', default=None, help='Optional single merged parquet output path (created after per-file processing)')
    p.add_argument('--sample_frac', type=float, default=None, help='If set, produce a sampled merged parquet with this fraction')
    p.add_argument('--stratified', action='store_true', help='If sampling, do stratified sampling per class')
    args = p.parse_args()
    main(args)
