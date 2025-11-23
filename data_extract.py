# name: Scott Landry

import os
import random
import time
import cv2

#GLOBAL VARIABLES
DATASET_PATH = './deepaction_videos'
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


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
        video_capture = cv2.VideoCapture(self.video_path)
        # check if the video was opened successfully
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames / sample_size
        frame_list = []
        for i in range(sample_size):
            frame_index = int(i * step)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_list.append(frame)
        video_capture.release()
        return frame_list
    
class Model_Data:
    # class to represent a collection of generation model videos.
    
    def __init__(self, model_index, model_path):
        self.model_index = model_index
        self.model_name = os.path.basename(model_path)
        self.training_videos = []
        self.validation_videos = []
        self.test_videos = []
        self.video_paths = self.assign_videos(model_path)
        
    
    def assign_videos(self, path = DATASET_PATH):
        
        random.seed(time.time())
        
        
        video_paths = []
        # assign the videos to the training, validation, and test sets
        for file in os.listdir(path):
            video_paths.append(os.path.join(path, file))
        
        # Create a copy and shuffle randomly
        shuffled_videos = video_paths.copy()
        random.shuffle(shuffled_videos)
        
        # Calculate split sizes
        total_videos = len(shuffled_videos)
        train_count = int(total_videos * TRAIN_SIZE)
        val_count = int(total_videos * VAL_SIZE)
        #test_count = remaining videos
        
        # Assign videos to splits using enumerate
        for i, video_path in enumerate(shuffled_videos):
            if i < train_count:
                self.training_videos.append(video_path)
            elif i < train_count + val_count:
                self.validation_videos.append(video_path)
            else:
                self.test_videos.append(video_path)
            
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
 
