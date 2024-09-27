import harp
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import os
from time import time
from aeon.io.reader import Reader

def load(reader: Reader, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(root.name)}_{reader.register.address}_*.bin"
    data = [reader.read(file) for file in glob(pattern)]
    return pd.concat(data)

def concat_digi_events(series_low: pd.DataFrame, series_high: pd.DataFrame) -> pd.DataFrame:
    """Concatenate seperate high and low dataframes to produce on/off vector"""
    data_off = ~series_low[series_low==True]
    data_on = series_high[series_high==True]
    return pd.concat([data_off, data_on]).sort_index()


def get_register_object(register_number, harp_board='h1'):
    
    h1_reader = harp.create_reader(f'harp_resources/h1-device.yml', epoch=harp.REFERENCE_EPOCH)
    h2_reader = harp.create_reader(f'harp_resources/h2-device.yml', epoch=harp.REFERENCE_EPOCH)
    reference_dict = {
        'h1': {
            32: h1_reader.Cam0Event,
            33: h1_reader.Cam1Event,
            38: h1_reader.StartAndStop,
            46: h1_reader.OpticalTrackingRead
        },
        'h2': {
            38: h2_reader.Encoder,
            39: h2_reader.AnalogInput,
            42: h2_reader.ImmediatePulses
        }
    }
    return reference_dict[harp_board][register_number]

def read_ExperimentEvents(path):
    filenames = os.listdir(path/'ExperimentEvents')
    filenames = [x for x in filenames if x[:16]=='ExperimentEvents'] # filter out other (hidden) files
    sorted_filenames = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in filenames])).sort_values()
    read_dfs = []
    try:
        for row in sorted_filenames:
            read_dfs.append(pd.read_csv(path/'ExperimentEvents'/f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
        return pd.concat(read_dfs).reset_index().drop(columns='index')
    except pd.errors.ParserError as e:
        filename = f"ExperimentEvents_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"
        print(f'Tokenisation failed for file "{filename}".\n')
        print(f'Exact description of error: {e}')
        print('Likely due to extra commas in the "Value" column of ExperimentEvents. Please manually remove and run again.')
        return None
    except Exception as e:
        print('Reading failed:', e)
        return None

def read_OnixDigital(path):
    filenames = os.listdir(path/'OnixDigital')
    filenames = [x for x in filenames if x[:11]=='OnixDigital'] # filter out other (hidden) files
    sorted_filenames = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in filenames])).sort_values()
    read_dfs = []
    for row in sorted_filenames:
        read_dfs.append(pd.read_csv(path/'OnixDigital'/f"OnixDigital_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
    return pd.concat(read_dfs).reset_index().drop(columns='index')

def read_OnixAnalogFrameCount(path):
    filenames = os.listdir(path/'OnixAnalogFrameCount')
    filenames = [x for x in filenames if x[:20]=='OnixAnalogFrameCount'] # filter out other (hidden) files
    sorted_filenames = pd.to_datetime(pd.Series([x.split('_')[1].split('.')[0] for x in filenames])).sort_values()
    read_dfs = []
    for row in sorted_filenames:
        read_dfs.append(pd.read_csv(path/'OnixAnalogFrameCount'/f"OnixAnalogFrameCount_{row.strftime('%Y-%m-%dT%H-%M-%S')}.csv"))
    return pd.concat(read_dfs).reset_index().drop(columns='index')

def read_OnixAnalogData(dataset_path, binarise=False):
    # https://github.com/neurogears/vestibular-vr/blob/benchmark-analysis/Python/vestibular-vr/analysis/round_trip.py
    # https://open-ephys.github.io/onix-docs/Software%20Guide/Bonsai.ONIX/Nodes/AnalogIODevice.html
    start_time = time()
    arrays_to_concatenate = []
    files_to_read = [x for x in os.listdir(dataset_path/'OnixAnalogData')]
    
    def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    
    # Sort the files based on the extracted number
    files_to_read.sort(key=extract_number)
    
    for filename in files_to_read:
        with open(dataset_path/'OnixAnalogData'/filename, 'rb') as f:
            photo_diode = np.fromfile(f, dtype=np.int16)

            try:
                photo_diode = np.reshape(photo_diode, (-1,12))
            except:
                print(f'ERROR: Cannot reshape loaded "{filename}" binary file into [-1, 12] shape. Continuing with non-reshaped data.')
            
            arrays_to_concatenate.append(photo_diode)

    photo_diode = np.concatenate(arrays_to_concatenate)
    
    if binarise:
        PHOTODIODE_THRESHOLD = 120
        photo_diode[np.where(photo_diode <= PHOTODIODE_THRESHOLD)] = 0
        photo_diode[np.where(photo_diode > PHOTODIODE_THRESHOLD)] = 1
        photo_diode = photo_diode.astype(bool)

    print(f'OnixAnalogData loaded in {time() - start_time:.2f} seconds.')

    return photo_diode

def read_OnixAnalogClock(dataset_path):
    start_time = time()
    arrays_to_concatenate = []
    files_to_read = [x for x in os.listdir(dataset_path/'OnixAnalogClock')]
    
    def extract_number(filename):
        return int(filename.split('_')[-1].split('.')[0])
    
    # Sort the files based on the extracted number
    files_to_read.sort(key=extract_number)
    
    for filename in files_to_read:
        with open(dataset_path/'OnixAnalogClock'/filename, 'rb') as f:
            clock_data = np.fromfile(f, dtype=np.uint64)
            arrays_to_concatenate.append(clock_data) 
    
    output = np.concatenate(arrays_to_concatenate)

    print(f'OnixAnalogClock loaded in {time() - start_time:.2f} seconds.')

    return output

def read_fluorescence(photometry_data_path):
    Fluorescence = pd.read_csv(photometry_data_path/'Fluorescence.csv', skiprows=1, index_col=False)
    Fluorescence = Fluorescence.drop(columns='Unnamed: 5')
    return Fluorescence

def read_fluorescence_events(photometry_data_path):
    Events = pd.read_csv(photometry_data_path/'Events.csv', skiprows=0, index_col=False)
    return Events

def load_register_paths(dataset_path):
    
    if not os.path.exists(dataset_path/'HarpDataH1') or not os.path.exists(dataset_path/'HarpDataH2'):
        raise FileNotFoundError(f"'HarpDataH1' or 'HarpDataH2' folder was not found in {dataset_path}.")
    h1_folder = dataset_path/'HarpDataH1'
    h2_folder = dataset_path/'HarpDataH2'
    
    h1_files = os.listdir(h1_folder)
    h1_files = [f for f in h1_files if f.split('_')[0] == 'HarpDataH1']
    h1_dict = {int(filename.split('_')[1]):h1_folder/filename for filename in h1_files}
    
    h2_files = os.listdir(h2_folder)
    h2_files = [f for f in h2_files if f.split('_')[0] == 'HarpDataH2']
    h2_dict = {int(filename.split('_')[1]):h2_folder/filename for filename in h2_files}
    
    print(f'Dataset {dataset_path.name} contains following registers:')
    print(f'H1: {list(h1_dict.keys())}')
    print(f'H2: {list(h2_dict.keys())}')
    
    return h1_dict, h2_dict

def load_registers(dataset_path):

    start_time = time()
    
    h1_dict, h2_dict = load_register_paths(dataset_path)
    
    h1_data_streams = {}
    for register in h1_dict.keys():
        data_stream = load(get_register_object(register, 'h1'), dataset_path/'HarpDataH1')
        if data_stream.columns.shape[0] > 1:
            for col_name in data_stream.columns:
                h1_data_streams[f'{col_name}({register})'] = data_stream[col_name]
        elif data_stream.columns.shape[0] == 1:
            h1_data_streams[f'{data_stream.columns[0]}({register})'] = data_stream
        else:
            raise ValueError(f"Loaded data stream does not contain supported number of columns in Pandas DataFrame. Dataframe columns shape = {data_stream.columns.shape}")
            
    h2_data_streams = {}
    for register in h2_dict.keys():
        data_stream = load(get_register_object(register, 'h2'), dataset_path/'HarpDataH2')
        if data_stream.columns.shape[0] > 1:
            for col_name in data_stream.columns:
                h2_data_streams[f'{col_name}({register})'] = data_stream[col_name]
        elif data_stream.columns.shape[0] == 1:
            h2_data_streams[f'{data_stream.columns[0]}({register})'] = data_stream[data_stream.columns[0]]
        else:
            raise ValueError(f"Loaded data stream does not contain supported number of columns in Pandas DataFrame. Dataframe columns shape = {data_stream.columns.shape}")
            
    print(f'Registers loaded in {time() - start_time:.2f} seconds.')
    
    return {'H1': h1_data_streams, 'H2': h2_data_streams}