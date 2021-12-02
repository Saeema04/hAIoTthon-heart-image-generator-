import os
import numpy as np
from  pathlib import Path
import pydicom, cv2
import matplotlib.pyplot as plt
import tkinter 

root = Path(os.getcwd())
data = root / "data"
videos, frCnt = {}, 0

def selectFrames(image_files, start=0, frNum=-1):
    global videos, frCnt
    numVideos = len(image_files)
    frCnt = [0] * len(image_files)
    for i, file in enumerate(image_files):
        im = pydicom.dcmread(file)
        videos[file] = im.pixel_array
        frCnt[i] = videos[file].shape[0]

    minCnt = 10000
    for i in range(numVideos):  minCnt = frCnt[i] if  frCnt[i] < minCnt else minCnt
    if frNum == -1 or frNum >= minCnt: frNum =  minCnt

    fig = plt.figure(0)
    plt.set_cmap('gray')
    plt.get_current_fig_manager()
    rc, cc = 1, 1
    if numVideos > 1 and numVideos <= 3: cc = 3
    if numVideos > 3: rc, cc = 2, 3
    if numVideos > 6:   return 0 # do not display
    for i in range(start, frNum, 2):
        for j in range(numVideos):
            plt.subplot(rc, cc, j+1), plt.imshow(videos[image_files[j]][i][:,:,0])
        ans = plt.waitforbuttonpress(10)
        plt.clf()
        if ans == False: break
    plt.close()
    return i

def save_images(fname, frNum):
    global videos

    imFile = Path(fname).name
    name = imFile.split(".")[0]
    name = name + '_' + str(frNum) + '.jpg'
    p = root / 'save' / name

    im = videos[fname][frNum]
    rval = cv2.imwrite(str(p), im[:,:,0])
    print(rval)

def read_image(path, num=0):
    global videos
    if len(videos) == 0:
        im = pydicom.dcmread(path)
        im = im.pixel_array
        videos[path] = im
    else:
        im = videos[path]
    if im.ndim == 4:    #  it is a video of frames for ultrasound images (RGB for each pixel)
        im = im[num]  # First frame of the video
    if im.ndim == 3:       # if it is a color image
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    r, c = im.shape[:2]
    if r > c:
        nr, nc = 256, int(c/r*256)
    else:
        nr, nc = int(r/c*256), 256

    im = cv2.resize(im, (nc, nr), interpolation=cv2.INTER_LINEAR)
    im = (im - im.min()) / (np.max(im)-np.min(im))
    im = np.uint8(im * 255)
    image = np.zeros(shape=(256, 256))
    image[:im.shape[0],:im.shape[1]] = im.copy()
    image = image / (np.max(image))
    ### gausian filter 3x3
    # image = cv2.GaussianBlur(image, (3,3), 0)
    ### gaussian filter 5x5
    # image = cv2.GaussianBlur(image, (3,3), 0)
    ### laplacian filter 3x3
    # image = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
    # image = cv2.convertScaleAbs(image)
    ### low pass filter
    # kernel = np.ones((3,3), np.uint8)
    # kernel = kernel / np.sum(kernel)
    # image = cv2.filter2D(image, -1, kernel)
    # image = cv2.convertScaleAbs(image)

    ### high pass filter
    # kernel = np.array([[0.0, -1.0, 0.0], 
    #                [-1.0, 4.0, -1.0],
    #                [0.0, -1.0, 0.0]])
    # kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
    # image = cv2.filter2D(image, -1, kernel)
    # image = cv2.convertScaleAbs(image)
    
    return image


def find_files():
    image_paths = []
    for file in os.listdir(data):
        if file.endswith('.txt'):
            continue
        image_paths.append(str(data / file))
    return image_paths

