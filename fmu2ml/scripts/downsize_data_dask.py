import pyarrow.parquet as pq
import os

def split_parquet(input_file, output_dir, fraction=0.1):
    """Split large file without loading all into memory"""
    
    
    # Get total rows without reading data
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    n_rows = int(total_rows * fraction)
    
    # Read only the first portion
    df = pq.read_table(input_file).slice(0, n_rows).to_pandas()
    
    df.to_parquet(output_dir, index=False)
    
    print(f"Saved first {n_rows:,} rows out of {total_rows:,} to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Downsize large parquet files using Dask.")
    parser.add_argument("--input_file", type=str, help="Path to the input parquet file.")
    parser.add_argument("--output_file", type=str, help="Path to save the downsized parquet file.")
    parser.add_argument("--sample_fraction", type=float, default=0.1,
                        help="Fraction of data to sample (default: 0.1 for 10%).")
    
    args = parser.parse_args()
    
    split_parquet(args.input_file, args.output_file, args.sample_fraction)
