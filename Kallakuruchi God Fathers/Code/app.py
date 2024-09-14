from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
import os

app = Flask(__name__)  # Corrected from _name_ to __name__
CORS(app)  # Enable CORS for cross-origin requests

# Load the pre-trained model when the server starts
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

@app.route('/generate-video', methods=['POST'])
def generate_video():
    # Get the prompt from the request
    data = request.json
    prompt = data.get('prompt', 'Penguin dancing happily')
    
    # Generate video frames
    num_iterations = 4  # Number of times to run the pipeline for more frames
    all_frames = []
    
    for _ in range(num_iterations):
        video_frames = pipe(prompt).frames[0]
        all_frames.extend(video_frames)
    
    # Export the frames to a video file
    video_path = export_to_video(all_frames)
    
    # Send the video file as a response
    return send_file(video_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from transformers import pipeline
from huggingface_hub import login
import cv2

# Step 1: Log in to Hugging Face using a valid token
login(token="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", add_to_git_credential=True)

# Step 2: Load GPT-2 model for text generation
text_gen_model = pipeline('text-generation', model='gpt2')

# Define the input text
text_input = "Darth Vader is surfing on waves"

# Generate text using GPT-2 (with truncation to handle max_length warning)
generated_text = text_gen_model(text_input, max_length=50, num_return_sequences=1, truncation=True)
print("Generated Text: ", generated_text)

# Step 3: Load the text-to-video diffusion model
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)

# Ensure you're using the correct scheduler for better inference
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable CPU offloading for models, useful if using a lower memory environment
pipe.enable_model_cpu_offload()

# Step 4: Generate video from the text prompt
video_frames = pipe(text_input, num_inference_steps=40, height=320, width=576, num_frames=24).frames

# Ensure all frames have the correct shape (3 channels)
# Convert grayscale frames to RGB if necessary
if len(video_frames[0].shape) == 2:  # If the frames are grayscale
    video_frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) for frame in video_frames]

# Step 5: Export the generated frames to a video file
video_path = export_to_video(video_frames)

# Print the path of the saved video
print(f"Video saved at: {video_path}")