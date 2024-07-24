import cv2
import concurrent.futures

input_video_path = '/home/ikharitonov/Desktop/20204321_343_6.avi'
output_video_path = '/home/ikharitonov/Desktop/20204321_343_6_LANCZOS_10x.avi'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open the input video.")
    exit()

# Get the properties of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

# Define the upscale factor
upscale_factor = 8

# Calculate the dimensions of the upscaled video
upscaled_width = frame_width * upscale_factor
upscaled_height = frame_height * upscale_factor

# Create a VideoWriter object to write the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (upscaled_width, upscaled_height))

# Check if the VideoWriter was initialized successfully
if not out.isOpened():
    print("Error: Could not open the output video for writing.")
    cap.release()
    exit()

# # Read and process each frame
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Upscale the frame using Lanczos interpolation
#     upscaled_frame = cv2.resize(frame, (upscaled_width, upscaled_height), interpolation=cv2.INTER_LANCZOS4)

#     # Write the upscaled frame to the output video
#     out.write(upscaled_frame)

# # Release the VideoCapture and VideoWriter objects
# cap.release()
# out.release()

def process_frame(frame):
    """Upscales a single frame using Lanczos interpolation."""
    return cv2.resize(frame, (upscaled_width, upscaled_height), interpolation=cv2.INTER_LANCZOS4)

# Read and process frames in parallel
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Use ThreadPoolExecutor to process frames in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    upscaled_frames = list(executor.map(process_frame, frames))

print('Done upscaling.')

# Write the upscaled frames to the output video
for upscaled_frame in upscaled_frames:
    out.write(upscaled_frame)

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()

print("Done writing video.")