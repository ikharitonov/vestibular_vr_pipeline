import os
import h5py
import pandas as pd
import numpy as np
from pathlib import Path


def load_h5_streams_to_dict(data_paths):
    '''
    Takes list of H5 file paths and, loads streams into dictionary, and save to dictionary named by mouse ID
    '''
    #dict to save streams:
    reconstructed_dict = {} 
    # File path to read the HDF5 file
    for input_file in data_paths:
        name = input_file.split('/')[-1][-7:-3] # Given that file name is of format: resampled_streams_2024-08-22T13-13-15_B3M6.h5 
        
        if not os.path.exists(input_file):
            print(f'ERROR: {input_file} does not exist.')
            return None
    
        # Open the HDF5 file to read data
        with h5py.File(input_file, 'r') as h5file:
            print(f'reconstructing streams for mouse {input_file.split('/')[-1][-7:-3]}, from session folder: {input_file.split('/')[-3]}')
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
                
        reconstructed_dict[name] = reconstructed_streams
        print(f'  --> {input_file.split('/')[-1][-7:-3]} streams reconstructed and added to dictionary \n')
            

    return reconstructed_dict


def read_ExperimentEvents(path):
    filenames = os.listdir(path/'ExperimentEvents')
    filenames = [x for x in filenames if x[:16]=='ExperimentEvents'] # filter out other (hidden) files
    date_strings = [x.split('_')[1].split('.')[0] for x in filenames] 
    sorted_filenames = pd.to_datetime(date_strings, format='%Y-%m-%dT%H-%M-%S').sort_values()
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


def add_experiment_events(data_dict, events_dict, mouse_info):
    # Iterate over each mouse key in the dictionaries
    for mouse_key in data_dict:
        # Retrieve the main and event DataFrames
        main_df = data_dict[mouse_key]
        event_df = events_dict[mouse_key]

        # Ensure both indices are sorted
        main_df = main_df.sort_index()
        event_df = event_df.sort_index()

        # Perform a merge_asof on the index to add 'Value' as 'ExperimentEvents' with backward matching
        merged_df = pd.merge_asof(
            main_df,
            event_df[['Value']],  # Only select the 'Value' column from event_df
            left_index=True,
            right_index=True,
            direction='backward',
            tolerance=0  # Adjust tolerance for matching on the index
        )

        # Rename the 'Value' column to 'ExperimentEvents'
        if 'ExperimentEvents' in merged_df.columns:
            merged_df['ExperimentEvents'] = merged_df.pop('Value')  # Replace existing column with the new 'Value' column
            print(f'Pre-existing ExperimentEvents column was replaced with new for {mouse_key}')
        else:
            merged_df = merged_df.rename(columns={'Value': 'ExperimentEvents'})  # Add new column
            print(f'Added new ExperimentEvents for {mouse_key}')

        # Add metadata from event_df
        merged_df['Experiment'] = event_df['experiment'].unique()[0]
        merged_df['Session'] = event_df['session'].unique()[0]

        # Add mouse ID, sex, and brain area
        mouse_info_name = mouse_key[:4]
        merged_df['mouseID'] = mouse_info_name
        merged_df['sex'] = mouse_info[mouse_info_name]['sex']
        merged_df['area'] = mouse_info[mouse_info_name]['area']

        # Update the dictionary with the merged DataFrame
        data_dict[mouse_key] = merged_df

    return data_dict

def add_no_halt_column(data_dict, events_dict):
    # Iterate over each mouse in the dictionaries
    for mouse_key in data_dict:
        main_df = data_dict[mouse_key]  # Large DataFrame
        event_df = events_dict[mouse_key]  # Small DataFrame

        # Ensure the index of the event_df is named 'Seconds' and is numeric (milliseconds)
        event_df.index.name = 'Seconds'

        # Create a new column 'No_halt' in the main_df
        main_df['No_halt'] = False

        # Filter the 'No halt' events from event_df
        no_halt_events = event_df[event_df['Value'] == 'No halt']

        # Use pd.merge_asof to match the nearest milliseconds from main_df index to event_df index
        merged_df = pd.merge_asof(
            main_df,
            no_halt_events[['Value']],  # Only bring in the 'Value' column where 'No halt' appears
            left_index=True,  # main_df has time in its index
            right_index=True,  # no_halt_events has time in its index (both in ms)
            direction='backward',  # Choose closest event on or before the timestamp
            tolerance=0.00005  # Match down to 4 decimals
        )

        # Explicitly convert 'Value' to string and fill NaN with 'False'
        main_df['No_halt'] = (merged_df['Value'].astype(str).fillna('') == 'No halt')

        # Update the dictionary with the modified DataFrame
        data_dict[mouse_key] = main_df

        print('No_halt events added to', mouse_key)

        # Verification
        event_len = len(events_dict[mouse_key].loc[events_dict[mouse_key].Value == 'No halt'])
        data_len = len(data_dict[mouse_key].loc[data_dict[mouse_key].No_halt == True])
        if event_len != data_len:
            print(f'For {mouse_key}, the number of actual no-halt events is {event_len} and the number of True values in the data now is {data_len}')
        
        if event_len == data_len:
            print(f'  Correct number of no-halt events for {mouse_key}\n')

    return data_dict


def add_block_columns(df, event_df):
    # Iterate through each index and event value in event_df
    prev_column = None  # Tracks the column currently being filled as True
    for idx, event in event_df['Value'].items():
        if 'block started' in event:
            print(event)
            # Create a new column in df, filling with False initially
            column_name = event.split()[0]+'_block'
            df[column_name] = False

            # If there was a previous column being filled as True, set it to False up to this point
            if prev_column is not None:
                df.loc[:idx, prev_column] = False

            # Set the new column to True starting from this index
            df.loc[idx:, column_name] = True
            prev_column = column_name  # Track the events

        elif 'Block timer elapsed' in event:
    
            # If there's a current active block, set its values to False up to this point
            if prev_column is not None:
                df.loc[idx:, prev_column] = False

                prev_column = None  # Reset current column tracker

    # Ensure that any remaining True blocks are set to False after their end
    #if current_column is not None:
     #   df.loc[:, current_column] = False
    for col in df:
        if 'block started' in col:
            df.rename({col: f'{col.split()[0]}_block'}, inplace = True)
    
    return df

def check_block_overlap(data_dict):
    for mouse, df in data_dict.items():
        # Choose columns that end with _block
        block_columns = df.filter(regex='_block')
        # Check if any row has more than one `True` at the same time in the `_block` columns
        no_overlap = (block_columns.sum(axis=1) <= 1).all()
        # Check if each `_block` column has at least one `True` value
        all_columns_true = block_columns.any().all()
        if no_overlap and all_columns_true:
            print(f'For {mouse}: No overlapping True values, and each _block column has at least one True value')
        elif no_overlap and not all_columns_true:
            print(f'Not all block columns contains True Values for {mouse}')
        elif not no_overlap and all_columns_true:
            print(f'There are some overlap between the blocks {mouse}')


def pooling_data(datasets):
    '''
    :param datasets: [list of datasets]
    :return: one dataframe with all data
    *IMPORTANT: This must be the location of preprocessed.csv files,in a folder named by the recording time,
    in a folder where 4 first letters gives mouse ID
    '''
    pooled_data = pd.concat(datasets)
    return pooled_data


dtype_dict = {'Seconds':np.float64, 
    '470_dfF':np.float64,
    'movementX':np.float64, 
    'movementY':np.float64,
    'event':bool,
    'ExperimentEvents':object,
    'Experiment':object,
    'Session':object,
    'mouseID':object,
    'sex':object,
    'area':object,
    'No_halt':bool,
    'LinearMismatch_block':bool,
    'LinearPlaybackMismatch_block':bool}



def filter_data(data, filters = []):
    '''
    :param data: pandas df of pooled data
    :param filters: list that refers to filterdict within function, defines relevant filters through column names and values
    :param dur: optional list of two values: [seconds before event start, seconds after event start]
    :return: pd df of data corresponding to the filters chosen
    '''
    filterdict = {
        'V2M': ['Area', 'V2M'], # V2M
        'V1': ['Area', 'V1'], # V1
        'female': ['Sex', 'F'], # female
        'male': ['Sex', 'M'], # male
        'B1M3': ['mouseID', 'B1M3'],
        'B1M5': ['mouseID', 'B1M5'],
        'B2M1': ['mouseID', 'B2M1'],
        'B2M4': ['mouseID', 'B2M4'],
        'B2M5': ['mouseID', 'B2M5'],
        'B2M6': ['mouseID', 'B2M6'],
        'B3M1': ['mouseID', 'B3M1'],
        'B3M2': ['mouseID', 'B3M2'],
        'B3M3': ['mouseID', 'B3M3'],
        'B3M4': ['mouseID', 'B3M4'],
        'B3M5': ['mouseID', 'B3M5'],
        'B3M6': ['mouseID', 'B3M6'],
        'B3M7': ['mouseID', 'B3M7'],
        'B3M8': ['mouseID', 'B3M8'],
        'halt': ['event', True],
        'not_halt': ['event', False],
        'day1': ['Session', 'day1'],
        'day2': ['Session', 'day1'],
        'MM': ['Experiment', 'MMclosed-open'],
        'MM_regular':['Experiment', 'MMclosed-and-Regular'],
        'open_block': ['LinearPlaybackMismatch_block', True],
        'closed_block': ['LinearMismatch_block', True],
    }
    filtered_df = data
    for filter in filters:
        try:
            colname = filterdict[filter][0]
            valname = filterdict[filter][1]
            filtered_df = filtered_df.loc[filtered_df[colname]==valname]
        except KeyError:
            print('KeyError: \n Ensure filters appear in dataset and in filterdict (can be added)',
                 '\n Dataset will be returned without this filter: ', filter)
        if len(filtered_df) == 0:
            print(f'There are no {filter} in the filtered dict')
    return filtered_df


def norm(x, min, max):
    normal = (x-min)/(max-min)
    return normal


def align_to_event_start(df, trace, event_col, range_around_event):
    
    trace_chunk_list = []
    bsl_trace_chunk_list = []
    event_index_list = []
    
    # Identify the start times for each event
    event_times = df.loc[df[event_col] & ~df[event_col].shift(1, fill_value=False)].index
    
    # Calculate the time range around each event
    before_0 = range_around_event[0]
    after_0 = range_around_event[1]
    
    # Calculate the target length of each chunk based on the sampling rate
    sampling_rate = 0.001
    target_length = int(((before_0 + after_0) / sampling_rate) + 1)  # Include both ends
    Index= pd.Series(np.linspace(-range_around_event[0], range_around_event[1], target_length)) # common index
    
    for event_time in event_times:
        
        # Determine the time range for each chunk
        start = event_time - before_0
        end = event_time + after_0
        
        # Extract the chunk from the trace column
        chunk = df[trace].loc[start:end]
        
        # Normalize the index to start at -before_0
        chunk.index = (chunk.index - chunk.index[0]) - before_0
        # Check if the chunk is shorter than the target length
        if len(chunk) < target_length:
            # Pad the chunk with NaN values at the end to reach the target length
            padding = pd.Series([np.nan] * (target_length - len(chunk)), index=pd.RangeIndex(len(chunk), target_length))
            chunk = pd.concat([chunk, padding])
            chunk.index = Index # Getting the same index as the others
        
        # Baseline the chunk
        baselined_chunk = baseline(chunk)
        
        # Append the chunk and baselined chunk to lists
        trace_chunk_list.append(chunk.values)
        bsl_trace_chunk_list.append(baselined_chunk.values)
        event_index_list.append(event_time)  # Store the event time for use in final column names

    # Convert lists of arrays to DataFrames
    trace_chunks = pd.DataFrame(np.column_stack(trace_chunk_list), columns=event_index_list)
    bsl_trace_chunks = pd.DataFrame(np.column_stack(bsl_trace_chunk_list), columns=event_index_list)

    # Set the index as the common time range index for each chunk
    trace_chunks.index = Index
    bsl_trace_chunks.index = Index
    
    return trace_chunks, bsl_trace_chunks



def baseline(chunk):
    # Select the slice between -1 and 0 (from 1 second before event to event start)
    baseline_slice = chunk.loc[-1:0]
    
    # Calculate the mean of the baseline slice
    baseline_mean = baseline_slice.mean()
    
    # Subtract the baseline mean from the entire chunk to baseline it
    baselined_chunk = chunk - baseline_mean
    
    return baselined_chunk


    
