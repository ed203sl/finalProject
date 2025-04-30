import pandas as pd
import os
import sys
import datetime
import numpy as np
import re
import wfdb
from wfdb.io import rdann
import scipy
import pywt
from scipy.signal import resample

TARGET_FS = 500
SEGMENT_DURATION = 10
TARGET_SEGMENT_LENGTH = TARGET_FS * SEGMENT_DURATION

SCORED_LABELS = ["164889003", "164890007", "6374002", "426627000", "733534002", "713427006", "270492004"	,"713426002","39732003"	, "445118002","164909002","251146004","698252002","426783006","284470004","10370003","365413008","427172004","164947007","111975006","164917005","47665007","59118001","427393009","426177001","427084000","63593006","164934002","59931005","17338001"]
LABEL_ABBREVIATIONS = ["AF", "AFL", "BBB", "Brady", "CLBBB", "CRBBB", "IAVB", "IRBBB", "LAD", "LAnFB", "LBBB", "LQRSV", "NSIVCB", "NSR", "PAC", "PR", "PRWP", "PVC", "LPR", "LQT", "QAb", "RAD", "RBBB", "SA", "SB", "STach", "SVPB", "TAb", "TInv", "VPB"]
NORMAL_MAD = 0.6745

# Retrieve scored SNOMED-CT code labels from WFDB record object - Unscored labels are dropped
# Parameters: Comments - 'Comments' property of a WFDB record
def get_scored_labels(comments):
    labels = comments[2].split("Dx: ")[1].split(",")
    labels = [i for i in labels if i in SCORED_LABELS]
    return labels

# Load all WFDB records from a directory
# Parameters:
# path - path to the record directory
def load_signals_labels(pth, max_records=None):
    s_dir = pd.Series(os.listdir(pth)).sort_values()
    s_dir = s_dir[s_dir.str.contains('.hea')].reset_index(drop=True).apply(lambda filename: filename[:-4])
    if max_records is not None:
        s_dir = s_dir[:max_records]
    df_records = s_dir.to_frame(name='record_name')
    df_records['record'] = df_records.record_name.apply(lambda x: wfdb.rdrecord(pth + '\\' + x))
    df_records['comments'] = df_records.record.apply(lambda x: x.comments)

    df_records['signal'] = df_records['record'].apply(lambda x: x.p_signal)
    df_records['record_length'] = df_records.record.apply(lambda x: x.sig_len)
    df_records['frequency'] = df_records.record.apply(lambda x: x.fs)
    df_records['labels'] = df_records.comments.apply(lambda x: get_scored_labels(x))

    df_records = df_records[df_records.labels.apply(lambda x: True if len(x)>0 else False)].reset_index(drop=True)

    df_records.drop(columns=['record', 'comments'], inplace=True)

    return df_records


# Denoise a single-lead ECG signal using Discrete Wavelet Transform
# Parameters:
# signal - full-length ECG Signal, must be 1-dimensional
# wavelet - mother wavelet object - Default is Daubechies-5
# lvl - Decomposition level - default is 9
def denoise_dwt(signal, wavelet='db5', lvl=6):
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=lvl)
    threshold = ((np.median(np.abs(coeffs[-1]) / NORMAL_MAD )) * np.sqrt(2 * np.log(len(signal))))
    for i in range(1, len(coeffs)):
      coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    reconstructed = pywt.waverec(coeffs, wavelet=wavelet)
    return reconstructed


# denoise_dwt_all: Multi-lead signal denoising via Discrete Wavelet Transform
def denoise_dwt_all(signal, wavelet='db5', lvl=9):
  sig = signal.copy()
  for lead in range(signal.shape[1]):
    signal_denoised = denoise_dwt(sig[:, lead], wavelet=wavelet, lvl=lvl)
    if len(signal_denoised) != sig.shape[0]:
      if len(signal_denoised) > sig.shape[0]:
        signal_denoised = signal_denoised[:sig.shape[0]]
      else:
        signal_denoised = np.pad(signal_denoised, (0, signal.shape[0] - len(signal_denoised)), mode='constant')

    sig[:, lead] = signal_denoised

  return sig


# Baseline wander removal & denoising via bandpass filtering
# signal - Multi-channel ECG signal
def remove_baseline_wander(signal, fs):
    nyquist = fs / 2
    lowcut = 0.5 / nyquist  # Lower cutoff frequency (0.5 Hz)
    highcut = 40 / nyquist  # Upper cutoff frequency (40 Hz)
    b, a = scipy.signal.butter(4, [lowcut, highcut], btype='band')
    # Apply zero-phase filtering using filtfilt
    filtered_signal = scipy.signal.filtfilt(b, a, signal, axis=0)
    filtered_signal = filtered_signal - np.mean(filtered_signal, axis=0)
    return filtered_signal


# Convert dataframe of ECG signal records to uniform data format, 10 seconds @ 500Hz
def make_uniform(df):
    new_samples = []
    for index, row in df.iterrows():
        seg_len = row.frequency * SEGMENT_DURATION
        if row.record_length < seg_len:
            # Zero-pad shorter record to become 10-seconds in duration
            pad_len = seg_len - row.record_length
            pad_begin = pad_len // 2
            pad_end = pad_len - pad_begin
            row.signal = np.pad(
                row.signal,
                ((pad_begin, pad_end), (0,0)),
                mode='constant'
            )
            row.record_length = seg_len
        num_splits = row.record_length // seg_len
        split_records = np.array_split(row.signal, num_splits, axis=0)
        for i, record in enumerate(split_records):
            if row.frequency != TARGET_FS:
                segment = resample(record, TARGET_SEGMENT_LENGTH, axis=0)
            else:
                segment = record[:TARGET_SEGMENT_LENGTH]

            new_samples.append({
                'record_name': f"{row.record_name}_{i}",
                'signal': segment,
                'labels': row.labels,
                'frequency': TARGET_FS,
                'segment_duration': SEGMENT_DURATION
            })

    return pd.DataFrame(new_samples)


# Code only runs if load_datasets.py is executed - Will not run if imported
# Loads dataset, performs data processing
# Correct usage: load_datasets.py DATASET_PATH [TARGET_PATH]
if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = ""
    target_path = "."
    
    if len(sys.argv) == 1:
        print("USAGE: load_datasets.py DATASET_PATH [TARGET_PATH]")
    data_path = current_path + "\\" + sys.argv[1]
    if len(sys.argv) == 3:
        target_path = current_path + "\\" + sys.argv[2]
    
    # Get source directory names for all datasets
    source_paths = os.listdir(data_path)
    
    # List of all subdirectories for all dataset sources
    print(data_path)
    print(source_paths)
    sub_directories = [os.listdir(data_path+"\\"+source) for source in source_paths]
    
    for i, src_path in enumerate(source_paths):
        print(f"Loading & Saving Dataset {src_path}...")
        sub_dirs = sub_directories[i]
        for sub_dir in sub_dirs:
            print(f"---- Loading sub-directory {sub_dir}...")
            full_path = data_path+"\\"+src_path+"\\"+sub_dir
            records = os.listdir(full_path)
            
            # Load records in sub-directory, denoise signals & convert records to uniform format
            data = load_signals_labels(full_path)
            data.signal = data.apply(lambda row: remove_baseline_wander(row.signal, fs=row.frequency), axis=1)
            data.signal = data.signal.apply(lambda x: denoise_dwt_all(x))
            data = make_uniform(data)
            
            # Save data to pickle file format
            data.to_pickle(f"{target_path}\\{src_path}_{sub_dir}.pkl")

    print(f"{'='*10} Data processing complete! {'='*10}")
            


            
                        
    
    
    