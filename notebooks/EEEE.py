import os
import pyarrow.parquet as pq
import pyarrow as pa

def merge_parquet_files(output_filename='database.parquet'):
    # Get a list of all .parquet files in the current directory
    parquet_files = [f for f in os.listdir() if f.endswith('.parquet')]

    if not parquet_files:
        print("No .parquet files found in the current directory.")
        return

    # Initialize a ParquetWriter with the schema of the first file
    first_file = parquet_files[0]
    first_table = pq.read_table(first_file)
    schema = first_table.schema
    with pq.ParquetWriter(output_filename, schema) as writer:
        # Write each file to the output Parquet file
        for file in parquet_files:
            table = pq.read_table(file)
            writer.write_table(table)

    print(f"All .parquet files have been merged into '{output_filename}'.")

if __name__ == "__main__":
    merge_parquet_files()
