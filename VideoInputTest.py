import os
import google.genai as genai
from dotenv import load_dotenv
import time
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
client = genai.Client()


def process_video_and_summarize(file_path):
    """
    Uploads a video file and asks a Gemini model to summarize it.
    This method is for all file sizes.
    """
    try:
        print(f"Uploading file: {file_path}...")
        uploaded_file = client.files.upload(file=file_path)
        print(f"File uploaded successfully with name: {uploaded_file.name}")

        print("Waiting for file to be processed...")
        while client.files.get(name=uploaded_file.name).state == "PROCESSING":
            print("File is still processing, waiting for 5 seconds...")
            time.sleep(5)
        if client.files.get(name=uploaded_file.name).state == "FAILED":
            raise ValueError("File processing failed.")
        print("File processing complete.")

        print("Generating summary...")
        
        # CHANGE SCRIPT IN CONTENTS ARRAY 

        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=[uploaded_file, "This video is a video of a basketball player shooting around. Can you count how many shots he makes, as well as misses, as well as field goal percentage?"]
        )
        print("Response received:")
        print(response.text)
        return response.text

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    file_name = "videoDataset/meshooting.mp4"
    process_video_and_summarize(file_name)