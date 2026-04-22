import pandas as pd
import numpy as np
import faiss
import os
from imblearn.over_sampling import SMOTE
from collections import Counter

TARGET_SAMPLES = 50
NUM_FRAMES = 20

def build_all_faiss_index(csv_file, index_file, label_file):
    df = pd.read_csv(csv_file)
    if df.empty:
        return False, "Data CSV kosong, rekam data manual dulu."

    all_X_final = []
    all_y_final = []
    
    vocab_groups = df.groupby('label')
    valid_vocabs = 0
    
    for vocab_name, group_df in vocab_groups:
        jml_sampel = len(group_df) // NUM_FRAMES
        # filter minimal 5 sampel biar smote ga error
        if jml_sampel < 5: 
            continue 
        
        valid_vocabs += 1
        grouped = group_df.groupby('video_id')
        
        # ambil array fitur nya
        X_raw = np.array([np.array([list(map(float, x.split(','))) for x in group['features']]).flatten() for _, group in grouped])
        
        # augmentasi geometris manual (noise + scale)
        X_aug = list(X_raw)
        for i in range(len(X_raw)):
            for _ in range(3): 
                noise = np.random.normal(0, 0.005, X_raw[i].shape)
                X_aug.append(X_raw[i] + noise)
            
            scale_factor = np.random.uniform(0.9, 0.95)
            X_aug.append(X_raw[i] * scale_factor) 
            
        X_aug = np.array(X_aug)
        y_aug = np.array([vocab_name] * len(X_aug))
        
        # smote butuh minimal 2 class, jadi kita akalin pake kelas dummy
        X_dummy = np.random.rand(*X_aug.shape)
        y_dummy = np.array(['dummy_class'] * len(X_aug))
        
        X_combined = np.vstack((X_aug, X_dummy))
        y_combined = np.concatenate((y_aug, y_dummy))
        
        k_neigh = min(4, len(X_aug) - 1)
        smote_strategy = {vocab_name: TARGET_SAMPLES, 'dummy_class': TARGET_SAMPLES}
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neigh)
        
        X_res, y_res = smote.fit_resample(X_combined, y_combined)
        
        # ambil yang vocab asli aja, dummynya buang
        all_X_final.append(X_res[y_res == vocab_name])
        all_y_final.append(y_res[y_res == vocab_name])
        
    if valid_vocabs == 0:
        return False, "Belum ada vocab yang nyampe 5 sampel."
        
    X_all_vocabs = np.vstack(all_X_final).astype('float32')
    y_all_vocabs = np.concatenate(all_y_final)
    
    # normalisasi wajid buat jarak cosine di faiss
    faiss.normalize_L2(X_all_vocabs)
    
    dimension = X_all_vocabs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(X_all_vocabs)
    
    faiss.write_index(index, index_file)
    np.save(label_file, y_all_vocabs)
    
    msg = f"Index FAISS sukses dibangun!\nTotal Vocab: {valid_vocabs}\nTotal Vektor: {len(y_all_vocabs)}"
    return True, msg