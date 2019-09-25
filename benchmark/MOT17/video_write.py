# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import subprocess


CODEC = "h264"  # "h264" or "mpeg4"

if __name__ == "__main__":

    frame_rates =  {
        "train": [30,30,14,30,30,30,25],
        "test": [30,30,14,30,30,30,25]
    }

    dir_names = {
        "train": [
            'MOT17-02-FRCNN',
            'MOT17-04-FRCNN',
            'MOT17-05-FRCNN',
            'MOT17-09-FRCNN',
            'MOT17-10-FRCNN',
            'MOT17-11-FRCNN',
            'MOT17-13-FRCNN',
        ],
        "test": [
            'MOT17-01-FRCNN',
            'MOT17-03-FRCNN',
            'MOT17-06-FRCNN',
            'MOT17-07-FRCNN',
            'MOT17-08-FRCNN',
            'MOT17-12-FRCNN',
            'MOT17-14-FRCNN',
        ]
    }

    cwd = os.getcwd()
    for mode in ["train", "test"]:
        for dir_name, frame_rate in zip(dir_names[mode], frame_rates[mode]):
            os.chdir(os.path.join(cwd, mode, dir_name, 'img1'))
            if CODEC == "h264":
                subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate), '-i', '%06d.jpg', '-c:v', 'libx264', '-f', 'rawvideo', '../{}-{}.mp4'.format(dir_name, CODEC)])
            elif CODEC == "mpeg4":
                subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate), '-i', '%06d.jpg', '-c:v', 'mpeg4', '-qscale:v', '1', '-f', 'rawvideo', '../{}-{}.avi'.format(dir_name, CODEC)])
