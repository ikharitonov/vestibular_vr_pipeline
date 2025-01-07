import harp
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import os
from time import time
from aeon.io.reader import Reader, Csv, Harp
import h5py
import json
from dotmap import DotMap
import aeon.io.api as api

class SessionData(Reader):
    """Extracts metadata information from a settings .jsonl file."""

    def __init__(self, pattern):
        super().__init__(pattern, columns=["metadata"], extension="jsonl")

    def read(self, file):
        """Returns metadata for the specified epoch."""
        with open(file) as fp:
            metadata = [json.loads(line) for line in fp] 

        data = {
            "metadata": [DotMap(entry['value']) for entry in metadata]
        }
        timestamps = [api.aeon(entry['seconds']) for entry in metadata]

        return pd.DataFrame(data, index=timestamps, columns=self.columns)


class TimestampedCsvReader(Csv):
    def __init__(self, pattern, columns):
        super().__init__(pattern, columns, extension="csv")
        self._rawcolumns = ["Time"] + columns

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["Seconds"] = data["Time"]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data
    

class PhotometryReader(Csv):
    def __init__(self, pattern):
        super().__init__(pattern, columns=["Time", "Events", "CH1-410", "CH1-470", "CH1-560", "U"], extension="csv")
        self._rawcolumns = self.columns

    def read(self, file):
        data = pd.read_csv(file, header=1, names=self._rawcolumns)
        data.set_index("Time", inplace=True)
        return data
    

class Video(Csv):
    def __init__(self, pattern):
        super().__init__(pattern, columns = ["HardwareCounter", "HardwareTimestamp", "FrameIndex", "Path", "Epoch"], extension="csv")
        self._rawcolumns = ["Time"] + self.columns[0:2]

    def read(self, file):
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["FrameIndex"] = data.index
        data["Path"] = os.path.splitext(file)[0] + ".avi"
        data["Epoch"] = file.parts[-3]
        data["Time"] = data["Time"].transform(lambda x: api.aeon(x))
        data.set_index("Time", inplace=True)
        return data



def load(reader: Reader, root: Path) -> pd.DataFrame:
    root = Path(root)
    pattern = f"{root.joinpath(reader.pattern).joinpath(reader.pattern)}_*.{reader.extension}"
    data = [reader.read(Path(file)) for file in glob(pattern)]
    return pd.concat(data)

def load_harp(reader: Harp, root: Path) -> pd.DataFrame:
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

def read_SessionSettings(dataset_path, print_contents=False):
    
    # File path to the jsonl file
    path = Path(dataset_path)
    jsonl_file_path = dataset_path/'SessionSettings'/os.listdir(dataset_path/'SessionSettings')[0]

    # Open and read the jsonl file line by line
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            # Parse each line as JSON
            data = json.loads(line)
            
            # Pretty print the data with correct indentation
            pretty_data = json.dumps(data, indent=4)

    if print_contents: print(pretty_data)

    return data

def read_fluorescence(photometry_data_path):
    try:
        Fluorescence = pd.read_csv(photometry_data_path/'Processed_Fluorescence.csv', skiprows=1, index_col=False)
    except FileNotFoundError:
        Fluorescence = pd.read_csv(photometry_data_path/'Fluorescence.csv', skiprows=1, index_col=False)
        
    if 'Unnamed: 5' in Fluorescence.columns: Fluorescence = Fluorescence.drop(columns='Unnamed: 5')
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
    
    # Converting any pd.DataFrames present (assumed to be single column DataFrames) into pd.Series
    for stream_name, stream in h1_data_streams.items():
        if type(stream) == pd.DataFrame:
            try:
                h1_data_streams[stream_name] = pd.Series(data=stream.values.squeeze(), index=stream.index)
            except:
                print(f'ERROR: Attempted to convert the loaded register "{stream_name}" to pandas.Series common format, but failed. Likely cause is that it has more than a single column.')
    for stream_name, stream in h2_data_streams.items():
        if type(stream) == pd.DataFrame:
            try:
                h1_data_streams[stream_name] = pd.Series(data=stream.values.squeeze(), index=stream.index)
            except:
                print(f'ERROR: Attempted to convert the loaded register "{stream_name}" to pandas.Series common format, but failed. Likely cause is that it has more than a single column.')
 
    print(f'Registers loaded in {time() - start_time:.2f} seconds.')
    
    return {'H1': h1_data_streams, 'H2': h2_data_streams}

def load_streams_from_h5(data_path):
    # File path to read the HDF5 file
    input_file = data_path/f'resampled_streams_{data_path.parts[-1]}.h5'

    if not os.path.exists(input_file):
        print(f'ERROR: {input_file} does not exist.')
        return None

    # Open the HDF5 file to read data
    with h5py.File(input_file, 'r') as h5file:
        # Read the common index (which was saved as Unix timestamps)
        common_index = h5file['HARP_timestamps'][:]
        
        # Convert Unix timestamps back to pandas DatetimeIndex
        # common_index = pd.to_datetime(common_index)
        
        # Initialize the dictionary to reconstruct the data
        reconstructed_streams = {}
        
        # Iterate through the groups (sources) in the file
        for source_name in h5file.keys():
            if source_name == 'HARP_timestamps':
                # Skip the 'common_index' dataset, it's already loaded
                continue
            
            # Initialize a sub-dictionary for each source
            reconstructed_streams[source_name] = {}
            
            # Get the group (source) and iterate over its datasets (streams)
            source_group = h5file[source_name]
            
            for stream_name in source_group.keys():
                # Read the stream data
                stream_data = source_group[stream_name][:]
                
                # Reconstruct the original pd.Series with the common index
                reconstructed_streams[source_name][stream_name] = pd.Series(data=stream_data, index=common_index)

    return reconstructed_streams