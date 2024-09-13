import cv2

def horizontal_flip_avi(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Define the codec and create a VideoWriter object for the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)  # 1 indicates horizontal flip

        # Write the flipped frame to the output video
        out.write(flipped_frame)

    # Release resources
    cap.release()
    out.release()

    print("Flipped video saved as:", output_video_path)