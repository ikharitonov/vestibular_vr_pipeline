from time import time
import numpy as np
import pandas as pd
from . import utils
import matplotlib.pyplot as plt
import copy
from datetime import timedelta
from datetime import datetime
import aeon.io.api as api

def resample_stream(data_stream_df, resampling_period='0.1ms', method='linear'):
    return data_stream_df.resample(resampling_period).last().interpolate(method=method)

def resample_index(index, freq):
    """Resamples each day in the daily `index` to the specified `freq`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The daily-frequency index to resample
    freq : str
        A pandas frequency string which should be higher than daily

    Returns
    -------
    pd.DatetimeIndex
        The resampled index

    """
    assert isinstance(index, pd.DatetimeIndex)
    start_date = index.min()
    end_date = index.max() + pd.DateOffset(days=1)
    resampled_index = pd.date_range(start_date, end_date, freq=freq)[:-1]
    series = pd.Series(resampled_index, resampled_index.floor('D'))
    return pd.DatetimeIndex(series.loc[index].values)

def get_timepoint_info(registers_dict, print_all=False):
    
    # Finding the very first and very last timestamp across all streams
    first_timestamps, last_timestamps = {}, {}
    for source_name, register_source in registers_dict.items():
        first_timestamps[source_name] = {k:v.index[0] for k,v in register_source.items()}
        last_timestamps[source_name] = {k:v.index[-1] for k,v in register_source.items()}
    
    # Saving global first and last timestamps across the sources and the registers
    joint_first_timestamps, joint_last_timestamps = [], []
    
    for register_source in first_timestamps.values():
        for register in register_source.values():
            joint_first_timestamps = joint_first_timestamps + [register]
    joint_first_timestamps = pd.DataFrame(joint_first_timestamps)
    
    for register_source in last_timestamps.values():
        for register in register_source.values():
            joint_last_timestamps = joint_last_timestamps + [register]
    joint_last_timestamps = pd.DataFrame(joint_last_timestamps)
    
    global_first_timestamp = joint_first_timestamps.iloc[joint_first_timestamps[0].argmin()][0]
    global_last_timestamp = joint_last_timestamps.iloc[joint_last_timestamps[0].argmax()][0]
    
    if print_all:
        print(f'Global first timestamp: {global_first_timestamp}')
        print(f'Global last timestamp: {global_last_timestamp}')
        print(f'Global length: {global_last_timestamp - global_first_timestamp}')
        
        for source_name in registers_dict.keys():
            print(f'\n{source_name}')
            for key in first_timestamps[source_name].keys():
                print(f'{key}: \n\tfirst  {first_timestamps[source_name][key]} \n\tlast   {last_timestamps[source_name][key]} \n\tlength {last_timestamps[source_name][key] - first_timestamps[source_name][key]} \n\tmean difference between timestamps {registers_dict[source_name][key].index.diff().mean()}')
    
    return global_first_timestamp, global_last_timestamp, first_timestamps, last_timestamps

def pad_and_resample(streams_dict, resampling_period='0.1ms', method='linear'):

    start_time = time()
    
    streams_dict = copy.deepcopy(streams_dict)
    
    # Padding: Getting the global first/last timepoints, adding them to every stream that starts/ends later/earler
    # Resampling + linear interpolation between points
    
    first_timestamp, last_timestamp, _, _ = get_timepoint_info(streams_dict)
    
    for source_name, source_element in streams_dict.items():
        for stream_name, stream in source_element.items():
            dummy_value = 0
            # Check if global first and last timestamps already exist in a given stream
            if stream.index[0] != first_timestamp:
                # Create new element with the earliest timestamp
                new_start = pd.Series([dummy_value], index=[first_timestamp])
                # Append the new element to the Series
                stream = pd.concat([new_start, stream])
                stream = stream.sort_index()
            if stream.index[-1] != last_timestamp:
                # Create new element with the latest timestamp
                new_end = pd.Series([dummy_value], index=[last_timestamp])
                # Append the new element to the Series
                stream = pd.concat([stream, new_end])
                stream = stream.sort_index()

            # Resampling and interpolation
            streams_dict[source_name][stream_name] = resample_stream(stream, resampling_period=resampling_period, method=method)
    
    print(f'Padding and resampling finished in {time() - start_time:.2f} seconds.')

    return streams_dict

def compute_Lomb_Scargle_psd(data_df, freq_min=0.001, freq_max=10**6, num_freqs=1000, normalise=True):
    freqs = np.linspace(freq_min, freq_max, num_freqs)
#     x = (data_df.index - data_df.index[0]).total_seconds().to_numpy()
    x = data_df.index
    y = data_df.values
    if y.ndim != 1: y = y[:,0]
    psd = signal.lombscargle(x, y, freqs, normalize=normalise)
    return freqs, psd

def plot_detail(data_stream_df, dataset_name, register, sample_num_to_plot=25):
    
    resampled_data_stream_df = resample_stream(data_stream_df)
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,15))
    
    fig.suptitle(f'DATASET [{dataset_name}] REGISTER [{register}]')
    
    ax[0][0].plot(data_stream_df, alpha=0.75)
    ax[0][0].plot(resampled_data_stream_df, alpha=0.75)
    ax[0][0].set_title('Full signal')
    ax[0][0].set_xlabel('Timestamp')
    ax[0][0].set_ylabel('Signal Magnitude')
    ax[0][0].set_xticklabels(ax[0][0].get_xticklabels(), rotation=-45)
    ax[0][0].grid()
    
    ax[1][0].plot(data_stream_df[:sample_num_to_plot], alpha=0.75)
    ax[1][0].scatter(data_stream_df[:sample_num_to_plot].index, data_stream_df[:sample_num_to_plot], s=25)
    filtered_resampled_df = resampled_data_stream_df[resampled_data_stream_df.index < data_stream_df.index[sample_num_to_plot]]
    ax[1][0].plot(filtered_resampled_df, alpha=0.75)
    ax[1][0].scatter(filtered_resampled_df.index, filtered_resampled_df, s=25, alpha=0.25)
    ax[1][0].set_xlabel('Timestamp')
    ax[1][0].set_ylabel('Signal Magnitude')
    ax[1][0].set_title(f'Zoom into first {sample_num_to_plot} timepoints')
    ax[1][0].set_xticks(data_stream_df[:sample_num_to_plot].index)
    ax[1][0].set_xticklabels(data_stream_df[:sample_num_to_plot].index.strftime('%H:%M:%S.%f'), rotation=-90)
    ax[1][0].grid()
    print('First five original timestamps:')
    for ts in data_stream_df[:5].index.to_list(): print(ts)
    print('\nFirst five resampled timestamps:')
    for ts in resampled_data_stream_df[:5].index.to_list(): print(ts)
    
    ax[1][1].plot(data_stream_df[-sample_num_to_plot:], alpha=0.75)
    ax[1][1].scatter(data_stream_df[-sample_num_to_plot:].index, data_stream_df[-sample_num_to_plot:], s=25)
    filtered_resampled_df = resampled_data_stream_df[resampled_data_stream_df.index >= data_stream_df.index[-sample_num_to_plot]]
    ax[1][1].plot(filtered_resampled_df, alpha=0.75)
    ax[1][1].scatter(filtered_resampled_df.index, filtered_resampled_df, s=25, alpha=0.25)
    ax[1][1].set_xlabel('Timestamp')
    ax[1][1].set_ylabel('Signal Magnitude')
    ax[1][1].set_title(f'Zoom into last {sample_num_to_plot} timepoints')
    ax[1][1].set_xticks(data_stream_df[-sample_num_to_plot:].index)
    ax[1][1].set_xticklabels(data_stream_df[-sample_num_to_plot:].index.strftime('%H:%M:%S.%f'), rotation=-90)
    ax[1][1].grid()
    
    inter_timestamp_invervals = np.diff(data_stream_df.index).astype(np.uint32) * (10**-9) # converted to seconds
    ax[2][0].hist(inter_timestamp_invervals, bins=50)
    ax[2][0].set_title('Histogram of intervals between timestamps')
    ax[2][0].set_xlabel('Inter-timestamp interval (seconds)')
    ax[2][0].set_ylabel('Count')
    ax[2][0].set_xticklabels(ax[2][0].get_xticklabels(), rotation=-45)
    ax[2][0].grid()
    
    plt.tight_layout()
    plt.show()

def plot_dataset(dataset_path):
    registers = utils.load_registers(dataset_path)
    h1_data_streams, h2_data_streams = registers['H1'], registers['H2']
    for register, register_stream in h1_data_streams.items():
        plot_detail(register_stream, dataset_path.name, register=str(register))
    for register, register_stream in h2_data_streams.items():
        plot_detail(register_stream, dataset_path.name, register=str(register))

def align_fluorescence_first_approach(fluorescence_df, onixdigital_df):
    # Aligns Fluorescence signal using the HARP timestamps from OnixDigital and interpolation
    # Steps:
    # - Selecting the rows where there are photometry synchronisation events occurring
    # - Getting the values from 'Seconds' column of OnixDigital and setting them to Fluorescence dataframe
    # - Estimating the very first and the very last 'Seconds' value based on timestamps of the photometry software ('TimeStamp' column)
    # - Applying default Pandas interpolation
    
    start_time = time()

    fluorescence_df = copy.deepcopy(fluorescence_df)
    
    # Adding a new column
    fluorescence_df['Seconds'] = np.nan
    
    # Setting the rows of Seconds column where there are events with HARP timestamp values from OnixDigital
    fluorescence_df.loc[fluorescence_df['Events'].notna(), 'Seconds'] = onixdigital_df['Seconds'].values
    
    # estimate the very first and very last values of Seconds column in Fluorescence to be able to interpolate between
    first_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['Seconds'] - fluorescence_df[fluorescence_df['Events'].notna()].iloc[0]['TimeStamp'] / 1000
    # first_val_to_insert = Seconds value of the first Event to occur - seconds elapsed since start of recording (converted from ms)
    last_val_to_insert = fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['Seconds'] + (fluorescence_df.iloc[-1]['TimeStamp'] / 1000 - fluorescence_df[fluorescence_df['Events'].notna()].iloc[-1]['TimeStamp'] / 1000)
    # last_val_to_insert = Seconds value of the last Event to occur + seconds elapsed between the last row of Fluorescence and the last event to occur
    
    fluorescence_df.loc[0, 'Seconds'] = first_val_to_insert
    fluorescence_df.loc[-1, 'Seconds'] = last_val_to_insert
    
    fluorescence_df[['Seconds']] = fluorescence_df[['Seconds']].interpolate()

    print(f'Fluorescence alignment finished in {time() - start_time:.2f} seconds.')
    
    return fluorescence_df

def convert_datetime_to_seconds(timestamp_input):
    if type(timestamp_input) == datetime or type(timestamp_input) == pd.DatetimeIndex:
        return (timestamp_input - utils.harp.REFERENCE_EPOCH).total_seconds()
    else:
        return timestamp_input.apply(lambda x: (x - utils.harp.REFERENCE_EPOCH).total_seconds())

def convert_seconds_to_datetime(seconds_input):
        return utils.harp.REFERENCE_EPOCH + timedelta(seconds=seconds_input)

def reformat_dataframe(input_df, name, index_column_name='Seconds', data_column_name='Data'):

    if input_df[index_column_name].values.dtype == np.dtype('<M8[ns]'):
        return pd.Series(data=input_df[data_column_name].values, 
                          index=input_df[index_column_name], 
                          name=name)
    else:
        return pd.Series(data=input_df[data_column_name].values, 
                            index=input_df[index_column_name].apply(convert_seconds_to_datetime), 
                            name=name)

def convert_arrays_to_dataframe(list_of_names, list_of_arrays):
    return pd.DataFrame({list_of_names[i]: list_of_arrays[i] for i in range(len(list_of_names))})

def convert_stream_from_datetime_to_seconds(stream):
    return pd.Series(data=stream.values, index=convert_datetime_to_seconds(stream.index))

def convert_all_streams_from_datetime_to_seconds(streams):
    for source_name in streams.keys():
        for stream_name in streams[source_name].keys():
            streams[source_name][stream_name] = convert_stream_from_datetime_to_seconds(streams[source_name][streams])
    return streams

def add_stream(streams, source_name, new_stream, new_stream_name):
    if not source_name in streams.keys():
        streams[source_name] = {}
        
    streams[source_name][new_stream_name] = new_stream
    
    return streams

def reformat_and_add_many_streams(streams, dataframe, source_name, stream_names, index_column_name='Seconds'):
    for stream_name in stream_names:
        new_stream = reformat_dataframe(dataframe, stream_name, index_column_name, data_column_name=stream_name)
        streams = add_stream(streams, source_name, new_stream, stream_name)
    return streams


def calculate_conversions_second_approach(data_path, photometry_path=None, verbose=True):

    start_time = time()
    output = {}

    OnixAnalogClock, OnixAnalogFrameCount = utils.read_OnixAnalogClock(data_path), utils.read_OnixAnalogFrameCount(data_path)

    # find time mapping/warping between onix and harp clock
    upsample = np.array(OnixAnalogFrameCount["Seconds"]).repeat(100, axis=0)[0:-100]

    if upsample.shape[0] != OnixAnalogClock.shape[0]:
        print('WARNING: "Unlucky" dataset with delayed subscription to OnixAnalogClock in Bonsai. As a consequence, the starting part of photodiode data is not counted. See https://github.com/neurogears/vestibular-vr/issues/81 for more information.')
        print(f'Shape of OnixAnalogClock == [{OnixAnalogClock.shape[0]}] shape of OnixAnalogFrameCount == [{upsample.shape[0]}].')
        print(f'Cutting {upsample.shape[0] - OnixAnalogClock.shape[0]} values from the beginning of OnixAnalogFrameCount. Data considered to be MISSING.')

        offset = upsample.shape[0] - OnixAnalogClock.shape[0]
        upsample = upsample[offset:]

    # define conversion functions between timestamps (onix to harp)
    o_m, o_b = np.polyfit(OnixAnalogClock, upsample, 1)
    onix_to_harp_seconds = lambda x: x*o_m + o_b
    onix_to_harp_timestamp = lambda x: api.aeon(onix_to_harp_seconds(x))
    harp_to_onix_clock = lambda x: (x - o_b) / o_m

    output["onix_to_harp_timestamp"] = onix_to_harp_timestamp
    output["harp_to_onix_clock"] = harp_to_onix_clock

    if photometry_path:
        OnixDigital = utils.read_OnixDigital(data_path)
        PhotometryEvents = utils.read_fluorescence_events(photometry_path)

        # define conversion functions between timestamps (onix to harp)
        m, b = np.polyfit(PhotometryEvents['TimeStamp'].values, OnixDigital["Value.Clock"], 1)
        photometry_to_onix_time = lambda x: x*m + b
        photometry_to_harp_time = lambda x: onix_to_harp_timestamp(photometry_to_onix_time(x))
        onix_time_to_photometry = lambda x: (x - b) / m
        
        output["photometry_to_harp_time"] = photometry_to_harp_time
        output["onix_time_to_photometry"] = onix_time_to_photometry

    if verbose:
        print('Following conversion functions calculated:')
        for k in output.keys(): print(f'\t{k}')
        print('\nUsage example 1: plotting photodiode signal for three halts')
        print('\n\t# Loading data')
        print('\tOnixAnalogClock = utils.read_OnixAnalogClock(data_path)\n\tOnixAnalogData = utils.read_OnixAnalogData(data_path)\n\tExperimentEvents = utils.read_ExperimentEvents(data_path)')
        print('\n\t# Selecting desired HARP times, applying conversion to ONIX time')
        print("\tstart_harp_time_of_halt_one = ExperimentEvents[ExperimentEvents.Value=='Apply halt: 1s'].iloc[0].Seconds\n\tstart_harp_time_of_halt_four = ExperimentEvents[ExperimentEvents.Value=='Apply halt: 1s'].iloc[3].Seconds")
        print("\tstart_onix_time = conversions['harp_to_onix_clock'](start_harp_time_of_halt_one - 1)\n\tend_onix_time = conversions['harp_to_onix_clock'](start_harp_time_of_halt_four)")
        print('\n\t# Selecting photodiode times and data within the range, converting back to HARP and plotting')
        print('\tindices = np.where(np.logical_and(OnixAnalogClock >= start_onix_time, OnixAnalogClock <= end_onix_time))')
        print("\tselected_harp_times = conversions['onix_to_harp_timestamp'](OnixAnalogClock[indices])\n\tselected_photodiode_data = OnixAnalogData[indices]")
        print('\tplt.plot(selected_harp_times, selected_photodiode_data[:, 0])')
        print('\nUsage example 2: plot photometry in the same time range')
        print('\n\tPhotometry = utils.read_fluorescence(photometry_path)')
        print("\n\tstart_photometry_time = conversions['onix_time_to_photometry'](start_onix_time)")
        print("\tend_photometry_time = conversions['onix_time_to_photometry'](end_onix_time)")
        print("\n\tselected_photometry_data = Photometry[Photometry['TimeStamp'].between(start_photometry_time, end_photometry_time)]['CH1-470'].values")
        print("\tselected_harp_times = conversions['photometry_to_harp_time'](Photometry[Photometry['TimeStamp'].between(start_photometry_time, end_photometry_time)]['TimeStamp'])")
        print('\tplt.plot(selected_harp_times, selected_photometry_data)')
        print("\nIt is best not to convert the whole OnixAnalogClock array to HARP timestamps at once (e.g. conversions['onix_to_harp_timestamp'](OnixAnalogClock)). It's faster to first find the necessary timestamps and indices in ONIX format as shown above.")


    print(f'Calculation of conversions finished in {time() - start_time:.2f} seconds.')

    return output

def select_from_photodiode_data(OnixAnalogClock, OnixAnalogData, hard_start_time, harp_end_time, conversions):

    start_time = time()

    start_onix_time = conversions['harp_to_onix_clock'](hard_start_time)
    end_onix_time = conversions['harp_to_onix_clock'](harp_end_time)
    indices = np.where(np.logical_and(OnixAnalogClock >= start_onix_time, OnixAnalogClock <= end_onix_time))

    x, y = conversions['onix_to_harp_timestamp'](OnixAnalogClock[indices]), OnixAnalogData[indices]

    print(f'Selection of photodiode data finished in {time() - start_time:.2f} seconds.')

    return x, y

def save_streams_to_excel(save_path, streams):

    start_time = time()

    source_names = [x for x in streams.keys()]
    source_dfs = {}
    
    for source_name in source_names:
        data = {}
        for stream_name in streams[source_name].keys():
            data[f'HARP_timestamps_{stream_name}'] = pd.DataFrame(streams[source_name][stream_name]).index
            data[stream_name] = pd.DataFrame(streams[source_name][stream_name]).values[:,0]
        source_dfs[source_name] = pd.DataFrame(data)

    # Save dataframes as excel files
    with pd.ExcelWriter(save_path) as writer:
        for source_name, source_df in source_dfs.items():
            source_df.to_excel(writer, sheet_name=source_name)

    print(f'Data exported as Excel file in {time() - start_time:.2f} seconds.')