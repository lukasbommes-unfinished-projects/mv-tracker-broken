# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import subprocess

DATASET = "train"  # "train" or "test"


if __name__ == "__main__":

    frame_rates = [30,30,30,30,30,30,14,14,14,30,30,30,30,30,30,30,30,30,25,25,25]

    if DATASET == "train":
        dir_names = [
            'MOT17-02-DPM',
            'MOT17-02-FRCNN',
            'MOT17-02-SDP',
            'MOT17-04-DPM',
            'MOT17-04-FRCNN',
            'MOT17-04-SDP',
            'MOT17-05-DPM',
            'MOT17-05-FRCNN',
            'MOT17-05-SDP',
            'MOT17-09-DPM',
            'MOT17-09-FRCNN',
            'MOT17-09-SDP',
            'MOT17-10-DPM',
            'MOT17-10-FRCNN',
            'MOT17-10-SDP',
            'MOT17-11-DPM',
            'MOT17-11-FRCNN',
            'MOT17-11-SDP',
            'MOT17-13-DPM',
            'MOT17-13-FRCNN',
            'MOT17-13-SDP',
        ]
    elif DATASET == "test":
        dir_names = [
            'MOT17-01-DPM',
            'MOT17-01-FRCNN',
            'MOT17-01-SDP',
            'MOT17-03-DPM',
            'MOT17-03-FRCNN',
            'MOT17-03-SDP',
            'MOT17-06-DPM',
            'MOT17-06-FRCNN',
            'MOT17-06-SDP',
            'MOT17-07-DPM',
            'MOT17-07-FRCNN',
            'MOT17-07-SDP',
            'MOT17-08-DPM',
            'MOT17-08-FRCNN',
            'MOT17-08-SDP',
            'MOT17-12-DPM',
            'MOT17-12-FRCNN',
            'MOT17-12-SDP',
            'MOT17-14-DPM',
            'MOT17-14-FRCNN',
            'MOT17-14-SDP',
        ]

    cwd = os.getcwd()
    for dir_name, frame_rate in zip(dir_names, frame_rates):
        os.chdir(os.path.join(cwd, DATASET, dir_name, 'img1'))
        subprocess.call(['ffmpeg', '-y', '-r', str(frame_rate), '-i', '%06d.jpg', '-c:v', 'libx264', '../seq.mp4'])
