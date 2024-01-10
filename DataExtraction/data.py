from frames import extract_frames
from visual import extract_visual_features
from audio import extract_audio_from_video, extract_audio_features_for_each_frame
from title import tfTitle
from save import saveData
from getData import getData
from visual import integrate_features
from objects import detectObjects

import sys
sys.path.append('videoSum')
from importances import cosine_distance

import spacy_sentence_bert
tokenizer = spacy_sentence_bert.load_model('en_stsb_roberta_large')

# import nltk
# from nltk.tokenize import sent_tokenize
# tokenizer=sent_tokenize

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import os

from frames import extract_frames

annotation_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-anno.tsv'
info_path='datasets/ydata-tvsum50-v1_1/data/ydata-tvsum50-info.tsv'
video_path='datasets/ydata-tvsum50-v1_1/video/'
summary_video_path='datasets/summary_videos/'
ground_truth_path='datasets/ydata-tvsum50-v1_1/ground_truth/ydata-tvsum50.mat'
video_list = [video for video in os.listdir(video_path) if video.endswith('.mp4')]  # List comprehension


def extractData(video_path, anno_file, info_file,flag_to_extract):
    
    return_data=[]
    # Extract frames from the video
    if(flag_to_extract[0]):
        frames = extract_frames(video_path)
        return_data.append(['frames',frames])
    else:
        return_data.append(None)

    # # Extract visual features
    if(flag_to_extract[1]):
        visual_features = extract_visual_features(frames) 
        return_data.append(['visual',visual_features])
    else:
        return_data.append(None)

    # Extract audio
    if(flag_to_extract[2]):
        audio_output_path = 'datasets/extractedAudio/extracted_audio.wav'
        extract_audio_from_video(video_path, audio_output_path) 

        # Extract audio features
        audio_features = extract_audio_features_for_each_frame(audio_output_path,30,len(frames))
        return_data.append(['audio',audio_features])
    else:
        return_data.append(None)

    # Load titles from info file
    if(flag_to_extract[3]):
        title_features = tfTitle(info_file,video_path,tokenizer)
        return_data.append(['title',title_features])
    else:
        return_data.append(None)


    return return_data


def score_frames_with_title_object(integrated_features, title_vector, object_vectors, bins=32):
    """Still not used"""
    frame_scores = []

    for i, frame_features in enumerate(integrated_features):

        # Calculate similarity or distance
        title_similarity = cosine_distance(frame_features, title_vector)
        sum_obj_sim=0
        for obj in object_vectors[i]:
            
            object_similarity = cosine_distance(frame_features, obj)
            sum_obj_sim+=object_similarity

        # Combine these similarities into a single score
        # This can be a simple average, weighted sum, or any other method that makes sense for your application
        combined_score = (title_similarity + sum_obj_sim) / 2

        frame_scores.append(combined_score)

    return frame_scores


def DataExtraction(video_path, anno_file, info_file,getDataFlag=False):
    """
    Integrate visual, audio, and annotation features from a video,
    and perform clustering on the combined features.

    :param video_path: Path to the video file.
    :param anno_file: Path to the annotation file.
    :param info_file: Path to the info file.
    :param num_clusters: Number of clusters to use in KMeans.
    :return: Cluster labels for each data point.
    """
    
    # Extract data from video
    objects=None
    
    video=video_path.split('/')[-1].split('.')[0]
    
    # GetData
    objects=getData('objects',video)
    frames=getData('frames',video)
    visual_features=getData('visual',video)
    audio_features=getData('audio',video)
    title_features=getData('title',video)
    encoded_objects=getData('encoded_objects',video)
    
    flag_to_extract=[True,True,True,True]
    if(frames is not None):
        flag_to_extract[0]=False
    if(visual_features is not None):
        flag_to_extract[1]=False
    if(audio_features is not None):
        flag_to_extract[2]=False
    if(title_features is not None):
        flag_to_extract[3]=False

    
    if not getDataFlag:
        # Extract data from video and save it
        data=extractData(video_path, anno_file, info_file,flag_to_extract)
        # Save extracted Data
        for d in data:
            if d is not None:
                if(d[0]=='objects'):
                    objects=d[1]
                elif(d[0]=='frames'):
                    frames=d[1]
                elif(d[0]=='visual'):
                    visual_features=d[1]
                elif(d[0]=='audio'):
                    audio_features=d[1]
                elif(d[0]=='title'):
                    title_features=d[1]
                
                saveData(d[0],d[1],video)
                
    if(objects is None):
        encoded_objects,objects = detectObjects(frames,len(visual_features),len(audio_features),encoded_objects=encoded_objects,video=video,tokenizer=tokenizer)
    else:
        encoded_objects,objects = detectObjects(frames,len(visual_features),len(audio_features),objects,encoded_objects=encoded_objects,tokenizer=tokenizer)
        
    saveData('encoded_objects',encoded_objects,video)
    
    return [encoded_objects,title_features]
