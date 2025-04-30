import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

training_sources = ['cpsc', 'ptb', 'incart', 'ningbo', 'chapman', 'georgia']
# Dataset Class
class ECGDataset(Dataset):
    # Possible domains - note that the Georgia source is never used during training or evaluation, only included here for compatibility with DataLoader.
    sources={'ningbo': 0, 'ptb': 1, 'chapman':2, 'incart':3, 'cpsc':4, 'georgia': 5}

    def __init__(self, data, normalize=False):
        self.data = data.reset_index(drop=True)
        self.signal_data = data.signal#.apply(lambda x: x.T)
        self.labels = data.labels_encoded
        self.normalize = normalize
        self.sources = self.sources


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record = self.data.iloc[idx]
        signal = record.signal
        signal = torch.tensor(record['signal'], dtype=torch.float32)
        if torch.isnan(signal).any():
            signal = torch.nan_to_num(signal, nan=0.0)    # Converts non-numeric values in signal (i.e. null values) to zero.
        signal = self.transform(signal) if self.transform else signal
        labels = torch.tensor(record['labels_encoded'], dtype=torch.float32)
        source_id = self.sources[record['source']]
        return signal, labels, source_id


def get_source(fname):
    for source in training_sources:
        if source in fname:
            return source

def load_dataset(training_files_path):
    data = pd.DataFrame()
    for i, training_file in enumerate(os.listdir(training_files_path)):
        df = pd.read_pickle(os.path.join(training_files_path, training_file))
        df['source'] = get_source(os.path.basename(training_file))
        data = pd.concat([data, df]).reset_index(drop=True)
    return data

def translate_labels(df, labels_translations_path):
    identical_classes = {'CLBBB': 'LBBB','CRBBB': 'RBBB','SVPB': 'PAC','VPB': 'PVC'}
    labels_translations = pd.read_csv(labels_translations_path)[["SNOMEDCTCode", "Abbreviation"]]
    #labels_list = labels_translations["Abbreviation"]
    labels_translations.SNOMEDCTCode = labels_translations.SNOMEDCTCode.astype(str)
    labels_translations = labels_translations.set_index("SNOMEDCTCode").to_dict()["Abbreviation"]
    df = df.explode('labels').replace({'labels': labels_translations}).replace({'labels': identical_classes}).reset_index(drop=False)
    df = df.groupby('index').aggregate({
        'signal':'first',
        'labels': list,
        'source':'first'
    })
    return df






# Runs if script is called rather than imported
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("USAGE: load_datasets.py create_datasets [PROCESSED_DATA_PATH] [LABEL_ABBREVIATIONS_PATH]")
        sys.exit(0)
    if not(sys.argv[1] == "create_datasets"):
        print("ERROR: Invalid arguments")
        print("USAGE: load_datasets.py create_datasets")
        sys.exit(-1)
    
    
    print("Creating train, validation & test data...")
    current_path = os.getcwd()
    #dir_files = os.listdir(os.path.join(current_path, os.path.join(current_path, sys.argv[2])))
    #all_files = [os.path.join(current_path, dir_file) for dir_file in dir_files]
    
    # Load dataset
    data = load_dataset(os.path.join(current_path, sys.argv[2]))
    data.signal = data.signal.apply(lambda x: x.T)
    data.reset_index(drop=True, inplace=True)
    
    # Calculate frequency of labels - since classes outside of top 20 are removed
    data = translate_labels(data, sys.argv[3])
    labels_frequency = data.labels.explode().value_counts().to_frame().reset_index(drop=False)
    labels_frequency["frequency"] = labels_frequency["count"].apply(lambda x: (f"{(x/len(data.labels)):.2%}"))
    top_20 = labels_frequency.labels[:20].values
    data = data[["signal", "labels", "source"]]
    data["labels"] = data["labels"].apply(lambda x: [label for label in x if label in top_20]) 
    data = data[data['labels'].apply(lambda x: len(x) > 0)] # Remove records with no remaining classes
    
    # Convert labels to binary encoding for appropriate input to model
    mlb = MultiLabelBinarizer(classes=top_20)
    labels_encoded = mlb.fit_transform(data['labels'])
    data['labels_encoded'] = list(labels_encoded)
    
    # Split dataframes to remove Georgia data
    data[data['source'].str.contains('georgia')].to_pickle('georgia_test_data.pkl')
    print("Georgia testing dataset created successfully..")
    data = data[~(data['source'].str.contains('georgia'))] # Remove Georgia records from data

    df_train, df_test = train_test_split(
    data[["signal", "labels", "labels_encoded", "source"]],
    random_state=41389703,
    test_size=0.2,
    shuffle=True
    )
    
    df_train.to_pickle(os.path.join(current_path,"training_data.pkl"))
    df_test.to_pickle(os.path.join(current_path, 'validation_data.pkl'))
    print("Training & Validation dataset created successfully. Created files are located in working directory.")
    
    
    
    
    
    
