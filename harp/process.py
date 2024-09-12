import os
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
import copy
from datetime import timedelta

def resample_stream(data_stream_df, resampling_period='0.1ms', method='linear'):
    return data_stream_df.resample(resampling_period).last().interpolate(method=method)

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
    
#     first_timestamps_h1 = {k:v.index[0] for k,v in stream_tuple[0].items()}
#     first_timestamps_h2 = {k:v.index[0] for k,v in stream_tuple[1].items()}
#     first_timestamps_other = {k:v.index[0] for k,v in stream_tuple[2].items()}
#     joint_first_timestamps = pd.DataFrame(list(first_timestamps_h1.values()) + list(first_timestamps_h2.values()) + list(first_timestamps_other.values()))
    
#     last_timestamps_h1 = {k:v.index[-1] for k,v in stream_tuple[0].items()}
#     last_timestamps_h2 = {k:v.index[-1] for k,v in stream_tuple[1].items()}
#     last_timestamps_other = {k:v.index[-1] for k,v in stream_tuple[2].items()}
#     joint_last_timestamps = pd.DataFrame(list(last_timestamps_h1.values()) + list(last_timestamps_h2.values()) + list(last_timestamps_other.values()))
    
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
        
#         print('\nH1:')
#         for key in first_timestamps_h1.keys():
#             print(f'{key}: \n\tfirst  {first_timestamps_h1[key]} \n\tlast   {last_timestamps_h1[key]} \n\tlength {last_timestamps_h1[key] - first_timestamps_h1[key]}')
        
#         print('\nH2:')
#         for key in first_timestamps_h2.keys():
#             print(f'{key}: \n\tfirst  {first_timestamps_h2[key]} \n\tlast   {last_timestamps_h2[key]} \n\tlength {last_timestamps_h2[key] - first_timestamps_h2[key]}')
        
#         print('\nOther:')
#         for key in first_timestamps_other.keys():
#             print(f'{key}: \n\tfirst  {first_timestamps_other[key]} \n\tlast   {last_timestamps_other[key]} \n\tlength {last_timestamps_other[key] - first_timestamps_other[key]}')
    
    return global_first_timestamp, global_last_timestamp, first_timestamps, last_timestamps

def pad_and_resample(streams_dict, resampling_period='0.1ms', method='linear'):
    
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
    
#     freq, psd = compute_Lomb_Scargle_psd(data_stream_df)
#     freq, psd_resampled = compute_Lomb_Scargle_psd(resampled_data_stream_df)
#     ax[0][1].plot(freq, psd, alpha=0.75)
#     ax[0][1].plot(freq, psd_resampled, alpha=0.75)
#     ax[0][1].set_title('Lomb Scargle periodogram')
#     ax[0][1].set_xlabel('Frequency')
#     ax[0][1].set_ylabel('Power Spectral Density')
#     ax[0][1].legend(['Original', 'Resampled'])
#     ax[0][1].grid()
    
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
    
    return fluorescence_df

def reformat_dataframe(input_df, name, index_column_name='Seconds', data_column_name='Data'):
    def convert_seconds_to_timestamps(seconds_input):
        return utils.harp.REFERENCE_EPOCH + timedelta(seconds=seconds_input)

    return pd.Series(data=input_df[data_column_name].values, 
                          index=input_df[index_column_name].apply(convert_seconds_to_timestamps), 
                          name=name)

def add_to_streams(streams, new_stream, new_stream_name):
    if not 'Non-HARP' in streams.keys():
        streams['Non-HARP'] = {}
        
    streams['Non-HARP'][new_stream_name] = new_stream
    
    return streams