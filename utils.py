from moviepy.editor import VideoFileClip
import os

def conv_mov_to_mp4(input_val, output_val, output_folder="videoDataset"):
    # create output folder if does not exist
    os.makedirs(output_folder, exist_ok=True)
    # Combine folder path with output filename
    output_path = os.path.join(output_folder, output_val)

    print(f"Converting {input_val} to .MP$4 format...")
    video_clip = VideoFileClip(input_val)
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    video_clip.close()
    print(f"Converted {input_val} to {output_val} successfully.")
input_file = "videoDataset/alivschisti2.MOV"
output_file = f"{input_file[13:-4]}.mp4"
conv_mov_to_mp4(input_file, output_file)


