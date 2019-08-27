# This script converts each sequence's imaged of the MOT dataset into H.264 encoded video sequences
import os
import glob
import cv2

DATASET = "test"  # "train" or "test"
PLAY_VIDEO = False  # whether to play video during conversion (slower)

def get_image_dims(filenames):
    try:
        probe_image = cv2.imread(filenames[0], cv2.IMREAD_COLOR)
        height, width, _ = probe_image.shape
        return width, height
    except IndexError:
        return None


if __name__ == "__main__":

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    
    frame_rates = [30, 30, 14, 30, 30, 30, 25,
                   30, 30, 14, 30, 30, 30, 25,
                   30, 30, 14, 30, 30, 30, 25]

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

    for dir_name, frame_rate in zip(dir_names, frame_rates):
        image_file_names = sorted(glob.glob(os.path.join(DATASET, dir_name, 'img1/*.jpg')))

        width, height = get_image_dims(image_file_names)

        video_file_name = os.path.join(DATASET, dir_name, 'seq.avi')
        print("Converting {}, fps={}, w={}, h={} ".format(video_file_name, frame_rate, width, height))
        video_writer = cv2.VideoWriter(video_file_name, fourcc, frame_rate, (width, height))

        # encode images to video
        for image_file_name in image_file_names:
            image = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
            video_writer.write(image)

            if PLAY_VIDEO:
                cv2.imshow("frame", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_writer.release()
