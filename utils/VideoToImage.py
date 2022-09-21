import cv2
import argparse
import os

def getFrame(sec,vidcap,count):
    
    global savedir
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(savedir,str(count)+".png"), image)     # save frame as JPG file
    return hasFrames

def ConvertImage(vidcap):
    
    sec = 0
    frameRate = 1 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec,vidcap,count)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec,vidcap,count)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Video to Image')
    parser.add_argument('-v', '--video', help='Video Path', required=True)
    parser.add_argument('-s', '--savedir', help='Save Directory', default = "./images")
    opt = parser.parse_args()
    savedir = opt.savedir
    if os.path.exists(savedir):
        files = os.listdir(savedir)
        for f in files:
            os.remove(os.path.join(savedir,f))
    else:
        os.makedirs(savedir, exist_ok=True)
    video = cv2.VideoCapture(opt.video)
    ConvertImage(video)