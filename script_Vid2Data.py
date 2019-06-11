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
#%% parameters
video_folder_path=r'D:\AI Data\DeepFake'
video_file_name='ZioNLight Bibi.mp4'

output_folder_path=r'D:\AI Data\DeepFake'
#output_resolution=(400,400) # scales according to the dimensions
output_resolution=0.3 # scales both axes according to this ratio, conceptually as dim1,dim2/=output_resolution
interpolation=cv2.INTER_AREA # see https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
output_jpg_quality=50 # 0<=int<=100, default: 95

#print_progress=True # prints each frame capturing
print_progress=False

#debug=True # breaking the extraction while loop after one iteration, for debugging
debug=False

#%% frame extraction
output_file_extension='jpg'
assert (isinstance(output_jpg_quality,int) and output_jpg_quality<=100 and output_jpg_quality>=0),'output_jpg_quality is not valid, should be 0<=int<=100!'

output_folder_path=os.path.join(output_folder_path,video_file_name+' frames')
if os.path.exists(output_folder_path):
    user_overwrite_approval=input('%s already exists, over-write all existing frames? y/[n] '%(output_folder_path))
    if user_overwrite_approval!='y':
        raise RuntimeError('aborted since the user disapproved over-writing!')
else:
    os.makedirs(output_folder_path)

# capturing frames (see https://www.geeksforgeeks.org/extract-images-from-video-in-python/)
video_file_path=os.path.join(video_folder_path,video_file_name)
vid_capture=cv2.VideoCapture(video_file_path) 

frame_i=0
while True:
    ret,frame=vid_capture.read()
    if frame_i==0:
        print(video_file_name+' original frame size: (%d,%d)'%(frame.shape[0],frame.shape[1]))
        if isinstance(output_resolution,tuple):
            output_resolution=(output_resolution[1],output_resolution[0])
        else:
            output_resolution=(round(frame.shape[1]*output_resolution),round(frame.shape[0]*output_resolution))
    
    if ret: # if a frame is still left
        frame=cv2.resize(frame,output_resolution,interpolation)  # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
        
        if frame_i==0:
            print('output frame size: (%d,%d)'%(frame.shape[0],frame.shape[1]))
        
        frame_name='frame %d.%s'%(frame_i,output_file_extension)
        frame_file_path=os.path.join(output_folder_path,frame_name)
        cv2.imwrite(frame_file_path,frame,
                    [cv2.IMWRITE_JPEG_QUALITY,int(output_jpg_quality)]) # https://www.programcreek.com/python/example/70397/cv2.IMWRITE_JPEG_QUALITY
        if frame_i==0:
            statinfo=os.stat(frame_file_path)
            frame_size_KB=statinfo.st_size/1000
            user_size_approval=input('1st output frame file size: %.2fKB, continue conversion of all frames? y/[n] '%(frame_size_KB))
            if user_size_approval!='y':
                raise RuntimeError('aborted since the user disapproved conversion for output frame file size')
        if print_progress:
            print(frame_name,'captured')       
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
