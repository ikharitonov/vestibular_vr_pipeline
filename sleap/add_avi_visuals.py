import cv2
import numpy as np
import load_and_process as lp

########### Data Loading ###########

# Loading and processing the points data
path = '/home/ikharitonov/Desktop/sleap_training/'
file = 'second_ellipse.003_20204321_343_5.analysis.csv'
df = lp.load_df(path+file)
columns_of_interest = ['left.x','left.y','center.x','center.y','right.x','right.y','p1.x','p1.y','p2.x','p2.y','p3.x','p3.y','p4.x','p4.y','p5.x','p5.y','p6.x','p6.y','p7.x','p7.y','p8.x','p8.y']
coordinates_dict = lp.get_coordinates_dict(df, columns_of_interest)
theta = lp.find_horizontal_axis_angle(df, 'left', 'right')
center_point = lp.get_left_right_center_point(coordinates_dict)
# columns_of_interest = ['left', 'right', 'center', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
columns_of_interest = ['left', 'right', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
remformatted_coordinates_dict = lp.get_reformatted_coordinates_dict(coordinates_dict, columns_of_interest)
# centered_coordinates_dict = lp.get_centered_coordinates_dict(remformatted_coordinates_dict, center_point)
# rotated_coordinates_dict = lp.get_rotated_coordinates_dict(remformatted_coordinates_dict, theta)
ellipse_parameters_data, ellipse_center_points_data = lp.get_fitted_ellipse_parameters(remformatted_coordinates_dict, ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8'])

# Transform points
ellipse_parameters_data = ellipse_parameters_data.reshape((ellipse_parameters_data.shape[0], 1, 3))
ellipse_center_points_data = ellipse_center_points_data.reshape((ellipse_center_points_data.shape[0], 1, 2))
# ellipse_center_points_data = lp.median_filter_smoothing(ellipse_center_points_data, 5)
pupil_points = np.swapaxes(np.array([arr for arr in remformatted_coordinates_dict.values()]), 0,1)
# pupil_points = lp.median_filter_smoothing(pupil_points, 5)


########### Video Editing ###########

def add_visuals_to_video(input_video_path, output_video_path, ellipse_parameters_data, ellipse_center_points_data, pupil_points):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Assuming you have a numpy array `points` of shape (num_frames, num_points, 2)
    # For example, points[0] contains coordinates for the first frame

    def draw_ellipse(frame, parameters, centers):
        for (width, height, phi), (center_x, center_y) in zip(parameters, centers):
            center = (int(center_x), int(center_y))
            axes = (int(width), int(height))
            angle = phi
            color = (0, 255, 0)
            thickness = 2

            # Draw ellipse
            cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness)

    def draw_points(frame, points, size = 1, color = (255,255,255)):
        for (x, y) in points:
            cv2.circle(frame, (int(x), int(y)), size, color, -1)

    progress_prints = {int(total_frames * (v/100)): f'{v}% of the video processed.' for v in [25, 50, 75, 100]}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < ellipse_parameters_data.shape[0]:
            draw_ellipse(frame, ellipse_parameters_data[frame_idx], ellipse_center_points_data[frame_idx])
            draw_points(frame, ellipse_center_points_data[frame_idx])
            draw_points(frame, pupil_points[frame_idx], size=2, color = (255, 0, 255))

        # Write the frame to the output video
        out.write(frame)

        if frame_idx in progress_prints.keys(): print(progress_prints[frame_idx])

        frame_idx += 1

    # Release video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path to input video and output video
input_video_path = '/home/ikharitonov/Desktop/20204321_343_5.avi'
output_video_path = '/home/ikharitonov/Desktop/20204321_343_5_POINTS.avi'

add_visuals_to_video(input_video_path, output_video_path, ellipse_parameters_data, ellipse_center_points_data, pupil_points)