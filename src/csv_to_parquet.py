import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time

def convert_csv_to_parquet():
    csv_file = 'dataset_dynamic.csv'
    parquet_file = 'dataset_dynamic.parquet'
    chunksize = 25000  # Cicil per 25.000 baris agar RAM sangat aman

    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} tidak ditemukan di direktori ini.")
        return

    print(f"🚀 Memulai konversi: {csv_file} -> {parquet_file}")
    print("Menggunakan sistem Chunking (Anti-OOM)...")
    
    start_time = time.time()
    writer = None
    row_count = 0

    try:
        # Membaca CSV secara dicicil
        for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunksize, low_memory=False)):
            # Konversi chunk DataFrame Pandas ke format Tabel PyArrow
            table = pa.Table.from_pandas(chunk)
            
            # Inisialisasi ParquetWriter pada iterasi pertama (untuk mengambil skema kolom)
            if writer is None:
                writer = pq.ParquetWriter(parquet_file, table.schema, compression='snappy')
            
            # Tambahkan data chunk ke dalam file Parquet
            writer.write_table(table)
            
            row_count += len(chunk)
            print(f"  -> Menyelesaikan chunk {i+1}... (Total baris sementara: {row_count:,})")

        # Tutup koneksi file
        if writer:
            writer.close()
            
        elapsed_time = time.time() - start_time
        print(f"\n✅ Konversi Berhasil dalam {elapsed_time:.2f} detik!")
        
        # Bandingkan ukuran file sebelum dan sesudah kompresi
        csv_size = os.path.getsize(csv_file) / (1024 * 1024)
        pq_size = os.path.getsize(parquet_file) / (1024 * 1024)
        
        print("\n📊 --- Laporan Efisiensi Penyimpanan ---")
        print(f"Ukuran CSV Asli     : {csv_size:.2f} MB")
        print(f"Ukuran Parquet Baru : {pq_size:.2f} MB")
        print(f"Rasio Kompresi      : Menghemat ruang hingga {100 - (pq_size/csv_size*100):.1f}%!")
        print("-----------------------------------------")
        print("Sekarang kamu bisa menghapus file .csv dan mulai menggunakan file .parquet")

    except Exception as e:
        print(f"\n❌ Terjadi error saat konversi: {e}")

if __name__ == "__main__":
    convert_csv_to_parquet()