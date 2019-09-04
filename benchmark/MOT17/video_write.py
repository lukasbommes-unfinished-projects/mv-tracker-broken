# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import subprocess

DATASET = "train"  # "train" or "test"
CODEC = "mpeg4"  # "h264" or "mpeg4"


if __name__ == "__main__":

    frame_rates =  {
        "train": [30,30,14,30,30,30,25],
        "test": [30,30,14,30,30,30,25]
    }

    dir_names = {
        "train": [
            'MOT17-02',
            'MOT17-04',
            'MOT17-05',
            'MOT17-09',
            'MOT17-10',
            'MOT17-11',
            'MOT17-13',
        ],
        "test": [
            'MOT17-01',
            'MOT17-03',
            'MOT17-06',
            'MOT17-07',
            'MOT17-08',
            'MOT17-12',
            'MOT17-14',
        ]
    }

    cwd = os.getcwd()
    for mode in ["train", "test"]:
        for dir_name, frame_rate in zip(dir_names[mode], frame_rates[mode]):
            os.chdir(os.path.join(cwd, mode, "{}-DPM".format(dir_name), 'img1'))
            if CODEC == "h264":
                subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate), '-i', '%06d.jpg', '-c:v', 'libx264', '../../../sequences/h264/{}.mp4'.format(dir_name)])
            elif CODEC == "mpeg4":
                subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate), '-i', '%06d.jpg', '-c:v', 'mpeg4', '-qscale:v', '1', '../../../sequences/mpeg4/{}.avi'.format(dir_name)])
