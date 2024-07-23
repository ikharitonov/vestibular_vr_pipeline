import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py

path = "/home/ikharitonov/Desktop/"
filename = "20204321_343_5.avi"
frames = []
cap = cv2.VideoCapture(path+filename)
ret = True
while ret:
    ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img)
video = np.stack(frames, axis=0) # dimensions (T, H, W, C)
# video = cv2.cvtColor(video,cv2.COLOR_BGR2GRAY)


# fnames = 'Sue_2x_3000_40_-46.tif'
# fnames = [download_demo(fnames)]     # the file will be downloaded if it doesn't already exist
# m_orig = cm.load_movie_chain(fnames)
downsample_ratio = .2  # motion can be perceived better when downsampling in time
# m_orig.resize(1, 1, downsample_ratio).play(q_max=99.5, fr=30, magnification=2)   # play movie (press q to exit)


max_shifts = (6, 6)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = False  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)

# if 'dview' in locals():
#     cm.stop_server(dview=dview)
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='multiprocessing', n_processes=None, single_thread=False)


# create a motion correction object
mc = MotionCorrect(video, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

# print(mc.mmap_file)

# mc.motion_correct(save_movie=True)

# load motion corrected movie
m_rig = cm.load('/home/ikharitonov/caiman_data/temp/tmp_mov_mot_corr.hdf5')
print(type(m_rig))
print(m_rig.shape)
# # bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
# #%% visualize templates
# plt.figure(figsize = (20,10))
# plt.imshow(mc.total_template_rig, cmap = 'gray')


# Writing
# cap = cv2.VideoCapture(0)
out = cv2.VideoWriter(path+filename.split('.')[0]+'_MC.avi', 1094862674, 20.0, (300,300))
i = 0

while(cap.isOpened()):
    # ret, frame = cap.read()
    if i < m_rig.shape[0]:
    # if ret==True:
    #     frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(m_rig[i])
        i += 1

        # cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


# with h5py.File('/home/ikharitonov/caiman_data/temp/tmp_mov_mot_corr.hdf5') as f:
#     corrected_movie = np.array(f['mov'])