# name: Scott Landry

import os
import random
import time
import cv2

import config


'''
This module is used to extract frames from a video and store them in a list.
it takes sample_size as an argument and returns that many frames evenly spaced from each other throughout the video.
the frames are stored in a list and can be accessed as a property of the Video class.
the frames are pulled lazily when the selected_frames property is accessed.



Stores videos in a list and assigns them to the training, validation, and test sets.
uses a seed to ensure that the videos are assigned to the training, validation, and test sets in the same way each time for reproducibility.

call the Model_Data class to get the training, validation, and test sets.

example: 
model = Model_Data(0, model_path)
training_videos = model.get_training_videos()
validation_videos = model.get_validation_videos()
test_videos = model.get_test_videos()
'''


class Video:
    # class to represent a single video file.
    
    def __init__(self, video_path, label_index, class_name, sample_size=10, pull_frames=False):
        self.video_path = video_path  #full path to video file
        self.label_index = label_index  # give the model a label index to identify the video
        self.class_name = class_name  # name of the video generation tool used to generate the video
        self.sample_size = sample_size  # store sample_size for lazy loading
            
        #initialize the selected frames to None until they are needed
        self._selected_frames = None
        if pull_frames:
            self.selected_frames  # trigger property to load frames
    
    # pull the frames once they are needed
    @property
    def selected_frames(self):
        if self._selected_frames is None:
            self._selected_frames = self.get_frames(self.sample_size)
        return self._selected_frames
    
    def get_frames(self, sample_size=10):
       #sample size is the number of frames to sample from the video evenly spaced throughout the video.
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")
        
        video_capture = cv2.VideoCapture(self.video_path)
        # check if the video was opened successfully
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames / sample_size
        frame_list = []
        for i in range(sample_size):
            frame_index = int(i * step)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # set the position of the video to the frame index
            ret, frame = video_capture.read() # read the frame
            if ret: # if the frame was read successfully, add it to the list
                frame_list.append(frame)
        video_capture.release()
        return frame_list
    
class Model_Data:
    # class to represent a collection of generation model videos.
    
    # model_index: index for this class (used as label)
    # model_path: path to the folder containing videos for this class
    # sample_size: number of frames to sample from each video
    def __init__(self, model_index, model_path, sample_size=10):
        self.model_index = model_index
        self.model_name = os.path.basename(model_path)
        self.sample_size = sample_size  # store sample_size to pass to Video objects
        self.training_videos = []
        self.validation_videos = []
        self.test_videos = []
        self.video_paths = self.assign_videos(model_path)
        
    
    def assign_videos(self, path):
        
        if hasattr(config, 'RANDOM_SEED'):
            random.seed(config.RANDOM_SEED)
        
        
        video_paths = []
        # assign the videos to the training, validation, and test sets
        for file in os.listdir(path):
            # only add video files
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
                video_paths.append(os.path.join(path, file))
        
        # Create a copy and shuffle randomly
        shuffled_videos = video_paths.copy()
        random.shuffle(shuffled_videos)
        
        # Calculate split sizes
        total_videos = len(shuffled_videos)
        train_count = int(total_videos * config.TRAIN_SIZE)
        val_count = int(total_videos * config.VAL_SIZE)
        #test_count = remaining videos
        
        # Assign videos to splits using enumerate - create Video objects
        for i, video_path in enumerate(shuffled_videos):
            # Create Video object (frames loaded lazily)
            video = Video(video_path, self.model_index, self.model_name, sample_size=self.sample_size)
            
            if i < train_count:
                self.training_videos.append(video)
            elif i < train_count + val_count:
                self.validation_videos.append(video)
            else:
                self.test_videos.append(video)
            
    def get_training_videos(self):
        return self.training_videos
    
    def get_validation_videos(self):
        return self.validation_videos
    
    def get_test_videos(self):
        return self.test_videos
    
    def get_size(self):
        return (len(self.training_videos) + len(self.validation_videos) + len(self.test_videos))
    
    def get_video_set(self, split):
        if split == 'training':
            return self.training_videos
        elif split == 'validation':
            return self.validation_videos
        elif split == 'test':
            return self.test_videos
        else:
            raise ValueError(f"Invalid split: {split}")
        
    def get_video_set_size(self, split):
        return len(self.get_video_set(split))
 