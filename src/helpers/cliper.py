import os
import cv2
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


class Cliper:
    def get_video_length(self, video_path):
        video = cv2.VideoCapture(video_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_length = total_frames / fps
        video.release()
        return video_length

    def download_from_yt(self, url, video_name, output_folder):
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        input_video_path = os.path.join(output_folder, f"{stream.default_filename}")
        # output_video_path = os.path.join(output_folder, f"{video_name}.mp4")
        stream.download(output_path=output_folder, filename=f"{video_name}.mp4")

    def create_clips(self, input_video_path, clip_duration=5, output_folder=""):
        video_length = self.get_video_length(input_video_path)
        total_clips = int(video_length // clip_duration)

        clips = []  # Store paths of the generated clips
        for i in range(total_clips):
            start_time = i * clip_duration
            end_time = (i + 1) * clip_duration
            clip_folder_name = f"output_clip_{i}"
            output_video_path = f"{output_folder}/{clip_folder_name}.mp4"

            ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=output_video_path)
            clips.append(output_video_path)

        return clips  # Return the list containing paths of the created clips

    def extract_frames(self, video_path, frame_rate, output_folder, x, start_time=None, end_time=None):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        video = cv2.VideoCapture(video_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        print(f"Frame rate: {frame_rate}, Total frames: {total_frames}, FPS: {fps}")
        frame_interval = x  # Set the frame interval to x
        print(f"Frame interval: {frame_interval}")

        # Calculate start and end frames
        start_frame = int(start_time * fps) if start_time is not None else 0
        end_frame = int(end_time * fps) if end_time is not None else total_frames

        count = 0
        saved_count = 0  # Separate counter for saved frame sequence
        while True:
            success, image = video.read()
            if not success or count > end_frame:
                break

            if count >= start_frame and count % frame_interval == 0:
                cv2.imwrite(
                    os.path.join(output_folder, f"{saved_count:05}.png"), image
                )  # Use saved_count for naming
                saved_count += 1

            count += 1

        video.release()

    def images_to_video(self, image_dir, fps, output_file):
        # Get all files from the directory
        files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ]

        # Sort the files by name
        files.sort()

        if not files:
            raise ValueError("No valid images found in the specified directory!")

        # Find out the frame size from the first image
        frame = cv2.imread(files[0])
        h, w, layers = frame.shape
        size = (w, h)

        # Define the codec using VideoWriter_fourcc and create a VideoWriter object
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"XVID"), fps, size)

        for file in files:
            img = cv2.imread(file)
            out.write(img)

        out.release()
        print(f"Video {output_file} is created successfully from images in {image_dir}")