from moviepy.editor import VideoFileClip
import os
import io
from moviepy.editor import VideoFileClip, concatenate_videoclips

from google.cloud import videointelligence_v1 as vi
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
#input_file = "videoDataset/alivschisti2.MOV"
#output_file = f"{input_file[13:-4]}.mp4"
#conv_mov_to_mp4(input_file, output_file)

# Function Detect objects in a video using Google Cloud Video Intelligence API
def detect_objects(path):
    """Detect objects (e.g., person, basketball) in a local video."""
    client = vi.VideoIntelligenceServiceClient()
    features = [vi.Feature.OBJECT_TRACKING]

    with io.open(path, "rb") as f:
        input_content = f.read()

    print("Sending video for object detection...")
    operation = client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
        }
    )

    result = operation.result(timeout=600)
    annotation_result = result.annotation_results[0]

    print("\n=== Detected Objects ===")
    segments = []

    for obj in annotation_result.object_annotations:
        name = obj.entity.description
        if name.lower() in ["person", "sports ball"]:
            start = obj.segment.start_time_offset.total_seconds()
            end = obj.segment.end_time_offset.total_seconds()
            print(f"{name}: {start:.2f}s - {end:.2f}s (confidence: {obj.confidence:.2f})")
            segments.append((start, end, name))

    return segments

def trim_video(input_path, output_path, segments):
    video = VideoFileClip(input_path)

    # Remove duplicates and sort segments
    unique_segments = sorted(set((s[0], s[1]) for s in segments))

    # Optional: merge overlapping/close segments
    final_segments = []
    buffer = 0.5  # seconds

    for start, end in unique_segments:
        if final_segments and start - final_segments[-1][1] < 1:
            # merge with previous
            final_segments[-1] = (final_segments[-1][0], max(final_segments[-1][1], end))
        else:
            final_segments.append((max(0, start - buffer), end + buffer))

    clips = [video.subclip(start, end) for start, end in final_segments]
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(output_path, codec="libx264")

test_file = "Game1SideA.mp4"
segments = detect_objects(f"videoDataset/{test_file}")
trim_video(f"videoDataset/{test_file}", "trimmed_video.mp4", segments)



