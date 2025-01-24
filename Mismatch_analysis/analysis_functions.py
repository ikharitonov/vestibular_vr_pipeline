#import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import os
import sys
from scipy.stats import sem

# Add the root directory to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
sys.path.append(project_root)

#from harp_resources import process, utils


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
    """
    Adds experimental events from events_dict to data_dict, aligning on the index and incorporating metadata.
    Critical events ('block started', 'Apply halt', 'No halt', 'Block timer elapsed') dominate and stand alone.
    """
    critical_events = {'block started', 'Apply halt', 'No halt'} #dict with the strings that must always be indcluded as single values

    for mouse_key in data_dict:
        # get main and event dfs
        main_df = data_dict[mouse_key]
        event_df = events_dict[mouse_key]

        #sort index
        main_df = main_df.sort_index()
        event_df = event_df.sort_index()

        #column to indicate if a critical event is present
        event_df['IsCritical'] = event_df['Value'].apply(
            lambda x: any(critical in x for critical in critical_events)
        )

        # Combine events, priority to critical events
        def combine_events(group):
            if group['IsCritical'].any():
                #if critical events exist, keep only those
                return ', '.join(sorted(group.loc[group['IsCritical'], 'Value'].unique()))
            else:
                # Otherwise, combine all events
                return ', '.join(sorted(group['Value'].unique()))

        event_df['CombinedEvents'] = event_df.groupby(event_df.index).apply(combine_events)

        # Drop duplicates and unnecessary columns
        event_df = event_df[['CombinedEvents', 'experiment', 'session']].drop_duplicates()

        # Perform a merge_asof on the index to add 'CombinedEvents' with backward matching
        merged_df = pd.merge_asof(
            main_df,
            event_df[['CombinedEvents']],
            left_index=True,
            right_index=True,
            direction='backward',
            tolerance=0  # Adjust tolerance if needed
        )

        #rename 'CombinedEvents' to 'ExperimentEvents'
        if 'ExperimentEvents' in merged_df.columns:
            merged_df['ExperimentEvents'] = merged_df.pop('CombinedEvents')
            print(f'Pre-existing ExperimentEvents column was replaced with new for {mouse_key}')
        else:
            merged_df = merged_df.rename(columns={'CombinedEvents': 'ExperimentEvents'})
            print(f'Added new ExperimentEvents for {mouse_key}')

        #add metadata to main df
        merged_df['Experiment'] = event_df['experiment'].iloc[0]
        merged_df['Session'] = event_df['session'].iloc[0]

        #Add mouse ID, sex, and brain area
        mouse_info_name = mouse_key[:4]
        merged_df['mouseID'] = mouse_info_name
        merged_df['sex'] = mouse_info[mouse_info_name]['sex']
        merged_df['area'] = mouse_info[mouse_info_name]['area']

        #Update the dict
        data_dict[mouse_key] = merged_df
        print(' dict updated\n')

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
            
def downsample_data(df, time_col='Seconds', interval=0.001):
    '''
    Uses pandas resample and aggregate functions to downsample the data to the desired interval. 
    * Note: Aggregation functions must be applied for each variable that is to be included.
    https://pandas.pydata.org/docs/reference/api/pandas.core.resample.Resampler.aggregate.html
    * Note: because the donwsampling keeps the first non-NaN value in each interval, some values could be lost.
    '''
    # Convert the Seconds column to a TimedeltaIndex
    df = df.set_index(pd.to_timedelta(df[time_col], unit='s'))

    #define aggregation functions for all possible columns
    aggregation_functions = {
        '470_dfF': 'mean', # takes the mean signal of the datapoints going into each new downsampled datapoint
        '560_dfF': 'mean',
        'movementX': 'mean',
        'movementY': 'mean',
        'event': 'any', # events column is a bool, and if there is any True values in the interval, the downsampled datapoint will be True
        'ExperimentEvents': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None, #first non-NaN value in the interval 
        'Experiment': 'first', # All values should be the same, so it can always just take the first string value
        'Session': 'first',
        'mouseID': 'first',
        'sex': 'first',
        'area': 'first',
        'No_halt': 'any', 
        'LinearMismatch_block': 'any', 
        'LinearPlaybackMismatch_block': 'any',
        'LinearRegular_block': 'any',
        'LinearClosedloopMismatch_block':'any',
        'LinearRegularMismatch_block':'any',
        'LinearNormal_block':'any',
    }

    # Filter aggregation_functions to only include columns present in df
    aggregation_functions = {key: func for key, func in aggregation_functions.items() if key in df.columns}

    # Resample with the specified interval and apply the filtered aggregations
    downsampled_df = df.resample(f'{interval}s').agg(aggregation_functions)

    # Reset the index to make the Seconds column normal again
    downsampled_df = downsampled_df.reset_index()
    downsampled_df[time_col] = downsampled_df[time_col].dt.total_seconds()  # Convert Timedelta back to seconds

    # Forward fill for categorical columns if needed, only if they exist in downsampled_df
    categorical_cols = ['Experiment', 'Session', 'mouseID', 'sex', 'area']
    for col in categorical_cols:
        if col in downsampled_df.columns:
            downsampled_df[col] = downsampled_df[col].ffill()

    # Remove consecutive duplicate values in the 'ExperimentEvents' column, if it exists
    if 'ExperimentEvents' in downsampled_df.columns:
        downsampled_df['ExperimentEvents'] = downsampled_df['ExperimentEvents'].where(
            downsampled_df['ExperimentEvents'] != downsampled_df['ExperimentEvents'].shift()
        )

    return downsampled_df


def test_event_numbers(downsampled_data, original_data, mouse):
    '''
    Counts number of True values in the No_halt columns in the original and the downsampled data
    This will indicate whether information was lost in the downsampling.
    If the original events somehow has been upsampled previously (for example if the tolerance was set too high in add_experiment_events()), 
    repeatings of the same event can also lead to fewer True events in the downsampled df.
    '''
    nohalt_down = len(downsampled_data.loc[downsampled_data['No_halt']==True])
    nohalt_original = len(original_data.loc[original_data['No_halt']==True])
    if nohalt_down != nohalt_original:
        print(f'mouse{mouse}')
        print(f'There are actually {nohalt_original} no-halts, but the downsampled data only contains {nohalt_down}')
        print('Should re-run the downsampling. Try changing interval lenght. Othewise, consider not downsampling\n')
    if nohalt_down == nohalt_original:
        print(f'mouse{mouse}')
        print(f'There are {nohalt_original} no-halts, and downsampled data contains {nohalt_down}\n')



def pooling_data(datasets):
    '''
    :param datasets: [list of datasets]
    :return: one dataframe with all data
    *IMPORTANT: This must be the location of preprocessed.csv files,in a folder named by the recording time,
    in a folder where 4 first letters gives mouse ID
    '''
    pooled_data = pd.concat(datasets)
    return pooled_data


#Define dtypes for efficient reading of csvs into pandas dataframe
#https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes
dtype_dict = {'Seconds':np.float64, 
    '470_dfF':np.float64,
    '560_dfF':np.float64,
    '410_dfF':np.float64,
    'z_470':np.float64,
    'z_410':np.float64,
    'z_560':np.float64,
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
    'LinearPlaybackMismatch_block':bool,
    'LinearRegularMismatch_block': bool,
    'LinearNormal_block': bool}



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
        'regular_block': ['LinearRegularMismatch_block', True],
        'normal_block': ['LinearNormal_block', True],
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
    """
    Align trace data around events with improved handling for trace chunks.
    """
    trace_chunk_list = []
    bsl_trace_chunk_list = []
    run_speed_list = []
    turn_speed_list = []
    event_index_list = []
    
    # Identify the start times for each event
    event_times = df.loc[df[event_col] & ~df[event_col].shift(1, fill_value=False)].index

    # Calculate the time range around each event
    before_0 = range_around_event[0]
    after_0 = range_around_event[1]
    
    # Calculate the target length of each chunk based on the sampling rate
    sampling_rate = 0.001
    target_length = int(((before_0 + after_0) / sampling_rate) + 1)  # Include both ends
    Index = pd.Series(np.linspace(-range_around_event[0], range_around_event[1], target_length))  # common index
 
    for event_time in event_times:
        # Determine the time range for each chunk
        start = event_time - before_0
        end = event_time + after_0
  
        # Extract the chunk from the trace column
        chunk = df[trace].loc[start:end]
        runspeed = df['movementX'].loc[start:event_time].mean()  # Saving mean run speed up until halt
        turningspeed = df['movementY'].loc[start:event_time].mean()
        
        # Normalize the index to start at -before_0
        chunk.index = (chunk.index - chunk.index[0]) - before_0
        
        # Check if the chunk is shorter than the target length
        if len(chunk) < target_length:
            # Pad the chunk with NaN values at the end to reach the target length
            padding = pd.Series([np.nan] * (target_length - len(chunk)), index=pd.RangeIndex(len(chunk), target_length))
            chunk = pd.concat([chunk, padding])
            chunk.index = Index  # Getting the same index as the others
    
        # Baseline the chunk
        baselined_chunk = baseline(chunk)
        
        # Append the chunk and baselined chunk to lists
        trace_chunk_list.append(chunk.values)
        bsl_trace_chunk_list.append(baselined_chunk.values)
        run_speed_list.append(runspeed)
        turn_speed_list.append(turningspeed)
        event_index_list.append(event_time)  # Store the event time for use in final column names
    
    if len(event_times) < 1:
        # Return empty DataFrames when there are no events
        trace_chunks = pd.DataFrame()
        bsl_trace_chunks = pd.DataFrame()
        movement_speeds = pd.DataFrame()
    else:
        # Convert lists of arrays to DataFrames
        trace_chunks = pd.DataFrame(np.column_stack(trace_chunk_list), columns=event_index_list)
        bsl_trace_chunks = pd.DataFrame(np.column_stack(bsl_trace_chunk_list), columns=event_index_list)
        run_speeds = pd.DataFrame(np.column_stack(run_speed_list), columns=event_index_list)
        turn_speeds = pd.DataFrame(np.column_stack(turn_speed_list), columns=event_index_list)
        movement_speeds = pd.concat([run_speeds, turn_speeds])
        
        # Set the index as the common time range index for each chunk
        trace_chunks.index = Index
        bsl_trace_chunks.index = Index
        movement_speeds.index = ['Mean_moveX', 'Mean_moveY']  # Set X and Y movement as movement speed index
        
    return trace_chunks, bsl_trace_chunks, movement_speeds



def baseline(chunk):
    #select slice between -1 and 0 (from 1 second before event to event start)
    baseline_slice = chunk.loc[-1:0]
    
    # Calculate the mean of the baseline slice
    baseline_mean = baseline_slice.mean()
    
    # Subtract the baseline mean from the entire chunk to baseline it
    baselined_chunk = chunk - baseline_mean
    
    return baselined_chunk


def view_session_mouse(mousedata_dict, mouse, plotlist=['470_dfF', 'movementX']):
    """
    Plot specified traces for a given mouse across sessions, including event-based highlighting and block annotations.
    
    Parameters:
    - mousedata_dict: Dictionary containing session data with DataFrames.
    - mouse: Mouse identifier (e.g., 'B3M3').
    - plotlist: List of column names to plot (e.g., ['470_dfF', 'movementX']).
    
    Returns:
    - fig, ax: Figure and axes of the generated plots.
    """
    print('\033[1m' + f'Plotted traces for {mouse}' + '\033[0m')

    # Determine the number of sessions and traces
    num_sessions = len(mousedata_dict)
    num_traces = len(plotlist)
    
    # Create subplots
    fig, ax = plt.subplots(
        num_traces, num_sessions,
        figsize=(15, 5 * num_traces), sharex=True
    )
    
    # Ensure `ax` is a 2D array for consistent indexing
    if num_traces == 1:
        ax = [ax]  # If only one trace, wrap in a list to index properly
    elif num_sessions == 1:
        ax = [[ax[i]] for i in range(num_traces)]  # Handle single session edge case

    # Loop through sessions and plot data for the specified mouse
    for s, (session, session_data) in enumerate(mousedata_dict.items()):
        # Filter data for the specified mouse
        mouse_data = session_data[session_data['mouseID'] == mouse]
        
        if mouse_data.empty:
            print(f"No data for {mouse} in session {session}")
            continue

        time = mouse_data.index
        event = mouse_data['halt']
        colors = plt.cm.tab10.colors  # Use a colormap to handle multiple traces

        for i, trace in enumerate(plotlist):
            if trace not in mouse_data.columns:
                print(f"Trace '{trace}' not found in session {session}")
                continue

            # Plot the trace
            ax[i][s].plot(time, mouse_data[trace], color=colors[i % len(colors)])
            ax[i][s].set_title(f"{trace} - {session}")
            ax[i][s].set_ylabel(trace)
            
            # Plot shaded areas for halt events
            ymin, ymax = ax[i][s].get_ylim()
            ax[i][s].fill_between(time, ymin, ymax, where=event, color='grey', alpha=0.3)

        # Plot annotations for block columns
        block_colors = ['lightsteelblue', 'lightcoral', 'forestgreen']
        colorcount = 0
        for col in mouse_data.columns:
            if '_block' in col and mouse_data[col].any():
                block_data = mouse_data[mouse_data[col]]
                start = block_data.index[0]
                end = block_data.index[-1]
                
                for i in range(num_traces):
                    ax[i][s].add_patch(Rectangle(
                        (start, ax[i][s].get_ylim()[0]), end - start, ax[i][s].get_ylim()[1] - ax[i][s].get_ylim()[0],
                        facecolor=block_colors[colorcount % len(block_colors)], alpha=0.1, edgecolor=None
                    ))
                
                # Annotate the block name below the x-axis
                ax[-1][s].text(
                    (start + end) / 2,  # Center of the block
                    ax[-1][s].get_ylim()[0] - 0.1 * (ax[-1][s].get_ylim()[1] - ax[-1][s].get_ylim()[0]),  # Position below x-axis
                    col, fontsize=10, verticalalignment='top', color=block_colors[colorcount % len(block_colors)],
                    fontweight='bold', ha='center'
                )
                colorcount += 1

    # Adjust layout
    fig.tight_layout(pad=2.0)
    plt.show()
    
    return fig, ax


        
def plot_compare_blocks(block_dict, event):
    # Determine number of blocks (columns) and maximum number of mice (rows)
    num_blocks = len(block_dict)
    max_mice = max(len(mice_data) for mice_data in block_dict.values())
    
    # Set up the figure with the determined number of rows and columns
    fig, ax = plt.subplots(max_mice, num_blocks, figsize=(5 * num_blocks, 3 * max_mice), squeeze=False)
    fig.suptitle(f'{event} alignment')
    
    # Dictionary to store mean data across mice for each block
    mean_mouse_dict = {block: {} for block in block_dict.keys()}
    
    # Loop over each block and each mouse, plotting down the rows within each block column
    for col, (block_name, mice_data) in enumerate(block_dict.items()):
        color_map = plt.cm.Greys  # Grey color map for traces
        
        # Loop over each mouse in the current block
        for row, (mouse, data) in enumerate(mice_data.items()):
            try:
                color = color_map(np.linspace(0, 1, data.shape[1]))  # Assign colors for traces
    
                # Plot vertical line for event alignment
                ax[row, col].axvline(x=0, linewidth=1, color='r', linestyle='--')
                
                # Plot individual traces with shading
                for idx, trace in enumerate(data.columns):
                    ax[row, col].plot(data.index, data[trace], color='grey', alpha=0.3)
    
                # Calculate mean and standard deviation across traces
                mean_trace = data.mean(axis=1)
                mean_mouse_dict[block_name][mouse] = mean_trace
                std_trace = data.std(axis=1)
    
                # Plot mean trace and standard deviation shading
                ax[row, col].plot(mean_trace, color='black', label='Mean' if row == 0 else "")
                ax[row, col].fill_between(mean_trace.index, mean_trace - std_trace, mean_trace + std_trace, alpha=0.3)
    
                # Add a shaded rectangle for a specified range (0 to 1)
                ax[row, col].add_patch(patches.Rectangle((0, ax[row, col].get_ylim()[0]), 1, 
                                                         ax[row, col].get_ylim()[1] - ax[row, col].get_ylim()[0], 
                                                         color='grey', alpha=0.1))
                # Set title and labels for the first row
                if row == 0:
                    ax[row, col].set_title(f"{block_name} responses")
                if col == 0:
                    ax[row, col].set_ylabel(f"Mouse: {mouse}")
            except AttributeError:
                pass
        

    fig.tight_layout(pad=1.08)

    # Aggregate means across mice for each block
    fig, ax = plt.subplots(1, num_blocks, figsize = (5 * num_blocks, 5))
    fig.suptitle('Mean across animal means')
    
    for col, (block_name, mean_data) in enumerate(mean_mouse_dict.items()):
        # Create DataFrame from mean data and compute overall mean and std across mice
        mean_df = pd.DataFrame.from_dict(mean_data)
        overall_mean = mean_df.mean(axis=1)
        overall_std = mean_df.std(axis=1)
        
        # Plot mean across animals with standard deviation shading
        ax[col].axvline(x=0, linewidth=1, color='r', linestyle='--')
        ax[col].plot(overall_mean, color='black')
        ax[col].fill_between(overall_mean.index, overall_mean - overall_std, overall_mean + overall_std, alpha=0.3)
        
        # Add rectangle to highlight the specified region (e.g., 0 to 1)
        ax[col].add_patch(patches.Rectangle((0, ax[col].get_ylim()[0]), 1, 
                                            ax[col].get_ylim()[1] - ax[col].get_ylim()[0], 
                                            color='grey', alpha=0.1))
        
        # Set title for each block
        ax[col].set_title(f'{block_name} loop mean response')

    return mean_mouse_dict



def extract_aligned_data(aligned_data_dict):
    # Initialize an empty list to store results
    results = []
    
    for session_number, session_blocks in aligned_data_dict.items():
        for session_block, mice_data in session_blocks.items():
            for mouse_id, item in mice_data.items():
                # Check if the item is a DataFrame
                if not isinstance(item, pd.DataFrame):
                    print(f"Warning: The data for Mouse ID '{mouse_id}' in session '{session_number}' and block '{session_block}' is not a DataFrame. Skipping.")
                    continue

                # Copy the DataFrame and ensure the index is numeric
                df = item.copy()
                df.index = pd.to_numeric(df.index)

                # Process each column independently
                for column in df.columns:
                    event_time_data = df.loc[0:1, column]  # Data during the event (0 to +1 seconds)
                    post_event_data = df.loc[1:2, column]  # Data during the first second after the event (+1 to +2 seconds)

                    peak_response = event_time_data.max()  # Max response during the event
                    min_response = event_time_data.min()  # Minimum response during the event
                    mean_response_event = event_time_data.mean()  # Mean response during the event
                    mean_response_post_event = post_event_data.mean()  # Mean response during the post-event time
                    min_response_post_event = post_event_data.min()  #Minimum response during the post-event time
                    peak_response_post_event = post_event_data.max() #Maximum response during the post-event time

                    #add results to list of dicts
                    results.append({
                        "SessionNumber": session_number,
                        "SessionBlock": session_block,
                        "MouseID": mouse_id,
                        "EventTime": column,
                        "PeakResponse": peak_response,
                        "MinResponse":  min_response,
                        "MeanResponse": mean_response_event,
                        "MeanResponse_after": mean_response_post_event,
                        "MinResponse_after": min_response_post_event,
                        "PeakResponse_after": peak_response_post_event
                    })

    # convert to a pandas df
    output_df = pd.DataFrame(results)
    return output_df



def align_to_event_start(df, trace_cols_to_baseline, trace_cols_no_baseline, event_col, range_around_event, sampling_rate=0.001):
    """
    Align multiple trace columns around events and apply baseline correction to specified traces.
    
    Parameters:
    df (pd.DataFrame): Input data.
    trace_cols_to_baseline (list): List of column names for traces to baseline (e.g., ['470_dfF', 'movementX']).
    trace_cols_no_baseline (list): List of column names for traces not to baseline.
    event_col (str): Column name marking events.
    range_around_event (list): Time range around the event [before, after].
    sampling_rate (float): Sampling rate in seconds (default 0.001).
    
    Returns:
    dict: Contains aligned trace chunks for each trace column.
    """
    aligned_chunks = {col: [] for col in trace_cols_to_baseline + trace_cols_no_baseline}  # Initialize a dictionary to store aligned chunks for each column
    event_index_list = []
    
    # Identify event times (when the event starts)
    event_times = df.loc[df[event_col] & ~df[event_col].shift(1, fill_value=False)].index
    
    # Time range to align around
    before_0, after_0 = range_around_event
    target_length = int(((before_0 + after_0) / sampling_rate) + 1)
    index_range = pd.Series(np.linspace(-before_0, after_0, target_length))
    
    for event_time in event_times:
        start, end = event_time - before_0, event_time + after_0
        
        # For each trace column, extract and align the chunk
        for trace_col in trace_cols_to_baseline:
            chunk = df[trace_col].loc[start:end]
            
            # Padding if the chunk is smaller than the target length
            if len(chunk) < target_length:
                padding = pd.Series([np.nan] * (target_length - len(chunk)), index=pd.RangeIndex(len(chunk), target_length))
                chunk = pd.concat([chunk, padding])
            
            # Baseline correction: subtract the mean of the first second prior to the event
            baseline_window = int(before_0 / sampling_rate)
            baseline_value = chunk.iloc[:baseline_window].mean()  # Mean of the first second before the event
            chunk = chunk - baseline_value  # Baseline correction
            
            chunk.index = index_range  # Align the index to the specified time range
            
            aligned_chunks[trace_col].append(chunk.values)
        
        # For traces without baseline correction, just align them
        for trace_col in trace_cols_no_baseline:
            chunk = df[trace_col].loc[start:end]
            
            # Padding if the chunk is smaller than the target length
            if len(chunk) < target_length:
                padding = pd.Series([np.nan] * (target_length - len(chunk)), index=pd.RangeIndex(len(chunk), target_length))
                chunk = pd.concat([chunk, padding])
            
            chunk.index = index_range  # Align the index to the specified time range
            
            aligned_chunks[trace_col].append(chunk.values)
        
        # Keep track of the event indices (for later indexing in the result)
        event_index_list.append(event_time)
    
    # If no event times were found, return empty DataFrames for all trace columns
    if len(event_times) < 1:
        return {col: pd.DataFrame() for col in trace_cols_to_baseline + trace_cols_no_baseline}
    
    # Add '_bsl' suffix to keys for baseline-corrected traces
    aligned_chunks = {
        (f'{col}_bsl' if col in trace_cols_to_baseline else col): aligned_chunks[col]
        for col in aligned_chunks
    }
    
    # Convert the list of aligned chunks into DataFrames for each trace column
    result = {col: pd.DataFrame(np.column_stack(aligned_chunks[col]), columns=event_index_list) for col in aligned_chunks}
    # Set the index to the time range
    for col in result:
        result[col].index = index_range
    
    return result

def process_mouse_data(mouse_data, trace_cols_to_baseline, trace_cols_no_baseline, event_col, range_around_event, sampling_rate=0.001):
    """
    Process mouse data dynamically across sessions, blocks, and mice, aligning multiple trace columns
    with baseline correction for specified columns.
    
    Parameters:
    mouse_data (dict): Nested dictionary of data.
    trace_cols_to_baseline (list): List of column names for traces to baseline (e.g., ['470_dfF', 'movementX']).
    trace_cols_no_baseline (list): List of column names for traces not to baseline.
    event_col (str): Column name marking events.
    range_around_event (list): Time range around the event [before, after].
    sampling_rate (float): Sampling rate in seconds (default 0.001).
    
    Returns:
    dict: Processed data containing aligned traces for each trace column.
    """
    aligned_data = {}
    
    # Iterate through the nested data structure (sessions, blocks, mice)
    for session, blocks in mouse_data.items():
        aligned_data[session] = {}
        for block, mice in blocks.items():
            aligned_data[session][block] = {}
            for mouse, df in mice.items():
                # Call align_to_event_start for each mouse's data, with baseline correction for specified traces
                aligned_result = align_to_event_start(df, trace_cols_to_baseline, trace_cols_no_baseline, event_col, range_around_event, sampling_rate)
                aligned_data[session][block][mouse] = aligned_result
    
    return aligned_data



def plot_compare_blocks(block_dict, trace):
    """
    Plot comparison across blocks for event-aligned data using baselined trace chunks.
    
    Parameters:
    - block_dict (dict): Nested dictionary with event-aligned data.
    - trace (str): The trace to align the traces, e.g., "bsl_trace_chunks".
    """
    # Determine number of blocks (columns) and maximum number of mice (rows)
    num_blocks = len(block_dict)
    max_mice = max(len(mice_data) for mice_data in block_dict.values())
    
    # Set up the figure with the determined number of rows and columns
    fig, ax = plt.subplots(max_mice, num_blocks, figsize=(5 * num_blocks, 3 * max_mice), squeeze=False)
    fig.suptitle("Comparison Across Blocks (Baselined Traces)", fontsize=16)
    
    # Iterate over blocks and mice to plot
    for col_idx, (block, mice_data) in enumerate(block_dict.items()):
        for row_idx, (mouse_id, mouse_data) in enumerate(mice_data.items()):
            # Extract the baselined trace chunks for the specified trace
            if trace in mouse_data:
                trace_data = mouse_data[trace]
                # Plot each trace
                for column in trace_data.columns:
                    ax[row_idx, col_idx].plot(trace_data.index, trace_data[column], c = 'grey', alpha = 0.2)
                    
                ax[row_idx, col_idx].plot(trace_data.index, trace_data.mean(axis=1), label=f'mean', c = 'blue', alpha = 0.5)
                ax[row_idx, col_idx].fill_between(
                trace_data.index,
                trace_data.mean(axis=1) + trace_data.std(axis=1),
                trace_data.mean(axis=1) - trace_data.std(axis=1),
                color='blue',
                alpha=0.3,
            )
            # Customize plot
            ax[row_idx, col_idx].set_title(f"Block: {block}, Mouse: {mouse_id}")
            ax[row_idx, col_idx].set_xlabel("Time (s)")
            ax[row_idx, col_idx].set_ylabel("Signal")
            ax[row_idx, col_idx].axvline()
            #ax[row_idx, col_idx].legend(loc="upper right", fontsize=8)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
    plt.show()



def mouse_mean_bsl_data(aligned_data, trace):
    """
    Extract mean traces from the 'bsl_trace_chunks' key for each mouse and restructure the dictionary.

    Parameters:
    - aligned_data (dict): Original dictionary with structure:
                           {session: block: mouse: 'trace_chunks': data, 'bsl_trace_chunks': data, ...}

    Returns:
    - processed_data (dict): Simplified dictionary with structure:
                             {session: block: mouse: mean_bsl_trace}
    """
    processed_data = {}

    for session, blocks in aligned_data.items():
        processed_data[session] = {}
        for block, mice in blocks.items():
            processed_data[session][block] = pd.DataFrame()
            for mouse, data in mice.items():
                if trace in data:
                    # Compute mean trace for 'bsl_trace_chunks'
                    bsl_data = data[trace]  # Assumes a 2D array (trials x time)
                    mean_bsl_trace = bsl_data.mean(axis=1) if bsl_data.size else np.nan
                    
                    processed_data[session][block][mouse] = mean_bsl_trace
    return processed_data

def plot_mean_across_blocks(session_blocks, control_blocks, title="Mean Across Blocks"):
    """
    Plot a single figure with the mean across mouse means for each block.
    
    Parameters:
    - session_blocks (dict): Dictionary of session data, where keys are block names
                             and values are DataFrames of aligned data.
    - control_blocks (dict): Dictionary of control data with the same structure.
    - title (str): Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.spines[['right', 'top']].set_visible(False)

    for block_name, mouse_data in session_blocks.items():
        if mouse_data.empty:
            print(f"No data for block: {block_name}")
            continue

        # Compute mean and std across mice
        mean_across_mice = mouse_data.mean(axis=1)
        std_across_mice = mouse_data.std(axis=1)

        # Plot block trace
        ax.plot(mean_across_mice, label=f'{block_name} Mean')
        ax.fill_between(
            mean_across_mice.index,
            mean_across_mice - std_across_mice,
            mean_across_mice + std_across_mice,
            alpha=0.2,
        )

    for block_name, mouse_data in control_blocks.items():
        if mouse_data.empty:
            print(f"No data for control block: {block_name}")
            continue

        # Compute mean and std across mice
        mean_across_mice = mouse_data.mean(axis=1)
        std_across_mice = mouse_data.std(axis=1)

        # Plot control block
        ax.plot(mean_across_mice, label=f'{block_name} Control Mean', color='black', linestyle='--')
        ax.fill_between(
            mean_across_mice.index,
            mean_across_mice - std_across_mice,
            mean_across_mice + std_across_mice,
            color='grey',
            alpha=0.3,
        )

    ax.axvline(0, color='grey', linestyle='--')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()


def compute_trace_statistics(
    data_dict,
    trace="470_dfF_bsl",
    co_traces=None,
    ranges=None,
):
    """
    Compute statistics for a main trace and optional co-traces, processing them simultaneously.

    Parameters:
    - data_dict: Nested dictionary with aligned data.
    - trace: Main trace for statistical computation.
    - co_traces: List of co-traces for computing mean values prior to the event.
    - ranges: List of time ranges for which to compute statistics. Defaults to [-1, 0], [0, 1], [1, 2].

    Returns:
    - pd.DataFrame: DataFrame with computed statistics.
    """
    if ranges is None:
        ranges = [(-1, 0), (0, 1), (1, 2)]  # Default time ranges

    if co_traces is None:
        co_traces = []  # Default to no co-traces

    result_rows = []

    for session, session_data in data_dict.items():
        for block_type, block_data in session_data.items():
            for mouse_id, mouse_data in block_data.items():
                # Ensure the main trace exists
                if trace not in mouse_data:
                    continue

                trace_chunks = mouse_data[trace]  # Main trace
                
                # Iterate through event columns
                for event_time in trace_chunks.columns:
                    row_base = {
                        "session": session,
                        "block_type": block_type,
                        "mouse_id": mouse_id,
                        "event_time": event_time,
                    }

                    # Compute statistics for co-traces in the [-1, 0] range
                    for co_trace in co_traces:
                        if co_trace in mouse_data:
                            co_trace_chunks = mouse_data[co_trace]
                            time_filtered_prior = co_trace_chunks.loc[-1:0, event_time]

                            row_base[f"{co_trace}_prior"] = (
                                time_filtered_prior.mean() if not time_filtered_prior.empty else np.nan
                            )
                        else:
                            row_base[f"{co_trace}_prior"] = np.nan

                    # Extract time ranges for the main trace
                    for start, end in ranges:
                        time_filtered = trace_chunks.loc[start:end, event_time]
                        row = row_base.copy()

                        if time_filtered.empty:
                            row.update({
                                f"time_range": f"{start}-{end}s",
                                "valid_data": False,
                                "peak": np.nan,
                                "mean": np.nan,
                                "median": np.nan,
                                "stderr": np.nan,
                            })
                        else:
                            # Compute statistics for the main trace
                            row.update({
                                f"time_range": f"{start}-{end}s",
                                "valid_data": True,
                                "peak": time_filtered.max(),
                                "mean": time_filtered.mean(),
                                "median": time_filtered.median(),
                                "stderr": sem(time_filtered, nan_policy="omit"),
                            })

                        result_rows.append(row)

    # Combine results into a single DataFrame
    result_df = pd.DataFrame(result_rows)

    # Ensure all co-trace columns are present even if missing
    for co_trace in co_traces:
        if f"{co_trace}_prior" not in result_df.columns:
            result_df[f"{co_trace}_prior"] = np.nan

    return result_df
