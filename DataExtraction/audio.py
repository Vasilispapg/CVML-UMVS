
from moviepy.editor import VideoFileClip
import librosa
import numpy as np

def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()
    
def extract_audio_features_for_each_frame(audio_path, frame_rate=2,num_frames=None):
    y, sr = librosa.load(audio_path)
    
    # frame_rate 1/1 = 30
    # frame_rate 1/2 = 15
    from keras.layers import Dense
    from keras.models import Sequential

    model = Sequential()
    model.add(Dense(1024, input_dim=128, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Calculate the number of audio samples per video frame
    samples_per_frame = sr // frame_rate
    
    # print("samples_per_frame",samples_per_frame)
    # print("len(y)",len(y))
    # print("sr",sr)
    # print("frame_rate",frame_rate)

    # Initialize an array to store MFCCs for each frame
    mfccs_per_frame = []

    # Iterate over each frame and extract corresponding MFCCs
    for frame in range(int(len(y) / samples_per_frame)):
        start_sample = int(frame * samples_per_frame)
        end_sample = int((frame + 1) * samples_per_frame)

        # Ensure the end sample does not exceed the audio length
        end_sample = min(end_sample, len(y))

        # Extract MFCCs for the current frame's audio segment
        mfccs_current_frame = librosa.feature.mfcc(y=y[start_sample:end_sample], sr=sr, n_mfcc=128)
        mfccs_processed = np.mean(mfccs_current_frame.T, axis=0)
        if mfccs_processed.ndim == 1:
            mfccs_processed = np.expand_dims(mfccs_processed, axis=0)


        expanded_features = model.predict(mfccs_processed)
        # print("expanded_features",expanded_features.shape)

        mfccs_per_frame.append(expanded_features[0])

    # print("len(mfccs_per_frame)",len(mfccs_per_frame))
    # print("len(mfccs_per_frame[0])",len(mfccs_per_frame[0]))
    
    if(len(mfccs_per_frame)>num_frames):
        return mfccs_per_frame[:num_frames]
    return mfccs_per_frame
