import subprocess
import glob
import os
import time
def main():
    videos = glob.glob("../videos/*.mp4")
    
    if not os.path.exists('../data'):
        os.mkdir('../data')

    for i, video in enumerate(videos):
        identity = video[-16:-6]
        try:
            os.mkdir('../data/{}'.format(identity))
        except FileExistsError as e:
            print("Not doing this as the files supposedly already exist.")
            continue
        subprocess.call(["../ffmpeg", "-i", video, '-vf', 
        'scale=256:256,fps=16', "-threads", "8", "-q:v", "3", '../data/{}/%07d.jpg'.format(identity)])
        time.sleep(5)
        print("File {0} out of {1} files extracted. Was file {2}.".format(i, len(videos), video))
        time.sleep(5)
main()


#':force_original_aspect_ratio=decrease,pad=288:288:-1:-1:color=black,setsar=1:1'