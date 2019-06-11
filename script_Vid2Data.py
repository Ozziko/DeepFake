# -*- coding: utf-8 -*-
"""@author: oz.livneh@gmail.com

* All rights of this project and my code are reserved to me, Oz Livneh.
* Feel free to use - for personal use!
* Use at your own risk ;-)

This script extracts data (currently only frames, in the future also audio) 
    from an input video, using OpenCV package (cv2 3.4.2)
"""
#%% imports
import cv2
import os
from time import time

class remainder_time:
    def __init__(self,time_seconds):
        self.time_seconds=time_seconds
        self.hours=int(time_seconds/3600)
        self.remainder_minutes=int((time_seconds-self.hours*3600)/60)
        self.remainder_seconds=time_seconds-self.hours*3600-self.remainder_minutes*60

#%% parameters
video_folder_path=r'D:\AI Data\DeepFake'
video_file_name='ZioNLight Bibi.mp4'

output_source_folder_path=r'D:\AI Data\DeepFake'
#output_resolution=(400,400) # scales according to the dimensions
output_resolution=0.6 # scales both axes according to this ratio, conceptually as dim1,dim2/=output_resolution
interpolation=cv2.INTER_AREA # see https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
output_jpg_quality=40 # 0<=int<=100, default: 95

print_progress_processed_frames_period=50 # prints progress every print_progress_processed_frames_period frames captured
#print_progress=0 # does not print progress

#debug=True # breaking the extraction while loop after one iteration, for debugging
debug=False

#%% reading video
output_file_extension='jpg'
assert (isinstance(output_jpg_quality,int) and output_jpg_quality<=100 and output_jpg_quality>=0),'output_jpg_quality is not valid, should be 0<=int<=100!'

# capturing frames (see https://www.geeksforgeeks.org/extract-images-from-video-in-python/)
video_file_path=os.path.join(video_folder_path,video_file_name)
vid_capture=cv2.VideoCapture(video_file_path) 

vic_fps=vid_capture.get(cv2.CAP_PROP_FPS)
vid_frames=vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
vid_duration_sec=vid_frames/vic_fps
vid_duration_obj=remainder_time(vid_duration_sec)

vid_height=vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
vid_width=vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
print('%s stats: duration: %d:%d:%d, fps: %.1f, frame size: (%d,%d)'%(
        video_file_name,
        vid_duration_obj.hours,
        vid_duration_obj.remainder_minutes,
        vid_duration_obj.remainder_seconds,
        vic_fps,
        vid_height,vid_width))

if isinstance(output_resolution,tuple):
    output_resolution=(output_resolution[1],output_resolution[0])
else:
    output_resolution=(round(vid_width*output_resolution),round(vid_height*output_resolution))

output_folder_path=os.path.join(output_source_folder_path,'%s %dx%d frames'%(
        video_file_name,output_resolution[1],output_resolution[0]))
if os.path.exists(output_folder_path):
    user_overwrite_approval=input('%s already exists, over-write all existing frames? y/[n] '%(output_folder_path))
    if user_overwrite_approval!='y':
        raise RuntimeError('aborted since the user disapproved over-writing!')
else:
    os.makedirs(output_folder_path)

#%% frame extraction
frame_i=0
tic=time()
while True:
    ret,frame=vid_capture.read()
    if ret: # if a frame is still left
        frame=cv2.resize(frame,output_resolution,interpolation)  # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/        
        frame_name='frame %d.%s'%(frame_i,output_file_extension)
        frame_file_path=os.path.join(output_folder_path,frame_name)
        cv2.imwrite(frame_file_path,frame,
                    [cv2.IMWRITE_JPEG_QUALITY,int(output_jpg_quality)]) # https://www.programcreek.com/python/example/70397/cv2.IMWRITE_JPEG_QUALITY
        if frame_i==0:
            statinfo=os.stat(frame_file_path)
            frame_size_KB=statinfo.st_size/1000
            user_size_approval=input('1st output frame file size: %.2fKB, expected total output: %.1fMB, continue conversion of all frames? y/[n] '%(
                    frame_size_KB,frame_size_KB*vid_frames/1e3))
            if user_size_approval!='y':
                raise RuntimeError('aborted since the user disapproved conversion for output frame file size')
        if print_progress_processed_frames_period>0 and frame_i%print_progress_processed_frames_period==print_progress_processed_frames_period-1:
            progress=(frame_i+1)/vid_frames
            passed_seconds=time()-tic
            expected_seconds=passed_seconds/progress*(1-progress)
            expected_remainder_time=remainder_time(expected_seconds)
                    
            print('extracted %.1f%% of frames, ETA: %dh:%dm:%.0fs'%(
                    100*progress,
                    expected_remainder_time.hours,
                    expected_remainder_time.remainder_minutes,
                    expected_remainder_time.remainder_seconds))       
        frame_i+=1
    else:
        break
    if debug:
        print('debug=True was set, breaking after one frame extraction for debugging')
        break
print('%d frames captured'%(frame_i))
# Release all space and windows once done 
vid_capture.release() 
cv2.destroyAllWindows() 
