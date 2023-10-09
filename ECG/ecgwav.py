import json
import numpy as np
import soundfile as sf
import subprocess
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import cv2

def change_pitch_ffmpeg(input_path, output_path, rate_multiplier):
    """Use ffmpeg to change pitch without changing speed."""
    cmd = [
    r'ffmpeg.exe', 
    '-i', input_path, 
    '-ar', '16000', 
    '-af', f'atempo=1/2,atempo=1/{rate_multiplier/2},asetrate={250*rate_multiplier},dynaudnorm=p=1:f=100', 
    '-y', 
    output_path
    ]
    subprocess.run(cmd)

def createVideoFromPngAndWavV1(pngFilePath, wavFilePath, outputVideoPath):
    """Take a PNG, add a WAV, make a video, it's a wrap!"""
    cmd = [
        r'ffmpeg.exe',
        '-loop', '1',                  # Loop the image
        '-i', pngFilePath,             # Input image file
        '-i', wavFilePath,             # Input audio file
        '-c:v', 'libx264',             # Video codec to use
        '-tune', 'stillimage',         # Optimize for still image input
        '-c:a', 'aac',                 # Audio codec to use
        '-strict', 'experimental',
        '-crf', '21', 
        '-preset', 'slow', 
        '-b:a', '128k',                # Audio bitrate
        '-shortest',                   # Finish encoding when the shortest input ends
        '-y', 
        '-pix_fmt', 'yuv420p',         # Pixel format, necessary for certain players
        '-threads', '16',
        outputVideoPath                # Output video file
    ]
    subprocess.run(cmd)

def createVideoFromPngAndWav(pngFilePath, wavFilePath, outputVideoPath):
    """Oh look, we're making movie magic!"""
    img = cv2.imread(pngFilePath)
    print(img.shape)
    height, width, _ = img.shape

    cmd = [
        r'ffmpeg.exe',
        '-loop', '1',
        '-i', pngFilePath,
        '-i', wavFilePath,
        '-vf', f'crop=2520:ih:(iw-2520)*t/30:0',
        '-c:v', 'libx264',
        '-r', '120',  # Setting the video frame rate to 60 fps
        '-tune', 'stillimage',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-crf', '21', 
        '-preset', 'slow', 
        '-b:a', '128k', 
        '-shortest', 
        '-y',
        '-pix_fmt', 'yuv420p',
        '-threads', '16',
        outputVideoPath
    ]
    subprocess.run(cmd)


# Read the JSON file
paths = [
    os.path.join(os.getcwd(), "FitBitECGLog.json"),
    os.path.join(os.getcwd(), "FitBitData", "FitBitECGLog.json"),
    os.path.join(os.getcwd(), "..", "FitBitData", "FitBitECGLog.json")
]
ecg_data = None
for path in paths:
    try:
        with open(path, 'r') as f:
            ecg_data = json.load(f)
            break
    except FileNotFoundError:
        continue
if ecg_data:
    print("FitBitECGLog.json data loaded successfully!")
else:
    print("Well, the elusive FitBitECGLog.json has eluded us once again. Touché, file... touché.")
    exit()



# Iterate through all ecgReadings
for i, ecg_reading in enumerate(ecg_data['ecgReadings']):
    waveform_samples = ecg_reading['waveformSamples']
    sampling_frequency = ecg_reading['samplingFrequencyHz']
    start_time = ecg_reading['startTime']
    averageHeartRate = ecg_reading['averageHeartRate']
    resultClassification = ecg_reading['resultClassification']
    est_ts = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
    est_ts_name = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d_%H-%M-%S")

    wav_file_path = f'ECG_{est_ts_name}_HR-{averageHeartRate}.wav'
    png_file_path_tmp = wav_file_path.replace('.wav','_tmp.png')
    png_file_path = wav_file_path.replace('.wav','.png')
    mp4_file_path = wav_file_path.replace('.wav','.mp4')

    if os.path.exists(mp4_file_path) and os.path.getsize(mp4_file_path) > 0:
        print('Already exists: ', est_ts, mp4_file_path)
        continue

    # Convert the waveform samples to a NumPy array and normalize
    waveform_np = np.array(waveform_samples, dtype=np.float32)
    waveform_np /= np.max(np.abs(waveform_np))

    # Create a time array for the x-axis
    time = np.arange(len(waveform_np)) / sampling_frequency

    # Create a waveform plot
    width = 2520/20
    height = 480/97.9
    fig, ax = plt.subplots(figsize=(width, height), dpi=100)  # Simple subplot method 


    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title(f'ECG Reading {est_ts} - Average Heart Rate: {averageHeartRate if averageHeartRate else "Inconclusive"} - Classification: {resultClassification}', color='white', fontsize=20)

    ax.tick_params(axis='x', colors='white')  # White ticks, because why not?
    ax.tick_params(axis='y', colors='white')

    # After plotting the data, before saving the plot:
    ax.set_xticks(np.arange(0, time[-1]+1, 1))  # This sets ticks from 0 to the last second, with an interval of 1 second
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:.0f}".format(x)))  # This ensures ticks are displayed as whole numbers

    # Set major ticks every 1 second.
    ax.set_xticks(np.arange(0, time[-1] + 1, 1))
    # Set minor ticks every 0.2 seconds.
    ax.set_xticks(np.arange(0, time[-1] + 1, 0.2), minor=True)

    # Enable both major and minor grids.
    ax.xaxis.grid(which='major', linestyle='-', linewidth='1', color='gray')
    ax.axhline(y=0, linestyle='-', linewidth='1', color='gray')
    
    ax.yaxis.grid(which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')


    ax.plot(time, waveform_np, color='white')  # A dashing white line for our ECG

    # spines
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    ax.set_facecolor('#004699')  # Deep, mysterious black for the plot background

    fig.tight_layout()

    ax.set_xlim(left=0, right=max(time))
    plt.savefig(png_file_path, bbox_inches='tight', facecolor='#004699')  # Ensure the background color is dark blue when saved

    #################
    # Save the plot as an image for the video!
    ax.yaxis.tick_right()  # Move the y-axis ticks to the right
    ax.yaxis.set_label_position("right")

    ax.set_xlim(left=-7.5,right=30)  # Make sure 0 is on the very left

    text_content = f'ECG Reading {est_ts}\nAverage Heart Rate: {averageHeartRate if averageHeartRate else "Inconclusive"}\nClassification: {resultClassification}\nDevice: {ecg_reading["deviceName"]}\nSampling Frequency (Hz): {ecg_reading["samplingFrequencyHz"]}\nLead Number: {ecg_reading["leadNumber"]}'
    x_position = -5  # Adjust this as needed for precise placement
    y_middle = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2  # Get the middle y-coordinate
    ax.text(x_position, y_middle, text_content, 
            verticalalignment='center', horizontalalignment='left',
            color='white', fontsize=25, bbox=dict(facecolor='#004699', edgecolor='white', boxstyle='round,pad=0.5'))


    current_position = ax.get_position()
    ax.set_position([0, current_position.y0, current_position.width, current_position.height])

    plt.savefig(png_file_path_tmp, bbox_inches='tight', facecolor='black')


    # Close the plot to free up resources
    plt.close()




    # Save as WAV file
    wav_tmp_file_path = f'ecg_sound_{i}.wav'
    sf.write(wav_tmp_file_path, waveform_np, sampling_frequency)

    # Change pitch using ffmpeg while maintaining the original speed
    change_pitch_ffmpeg(wav_tmp_file_path, wav_file_path, 4)
    os.remove(wav_tmp_file_path)

    createVideoFromPngAndWav(png_file_path_tmp, wav_file_path, mp4_file_path)
    os.remove(png_file_path_tmp)

    #exit()

