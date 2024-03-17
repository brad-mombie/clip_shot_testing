import cv2
import torch
from moviepy.editor import VideoFileClip
from pathlib import Path
import logging



def hello_world():
    print("Hello world")
    return

def process_video(video_path, skip_frames=5, buffer_before=3, buffer_after=1, detection_drop_threshold=5, debug=False):
    # At the beginning of your process_video function
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logging.info(f"Starting video processing for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    attempts = []
    current_attempt = []
    frame_idx = 0
    clip_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("Break")
            break

        if frame_idx % skip_frames == 0:
            results = model(frame)
            detections = results.xyxy[0]
            detected, confidence = rider_detected(detections, frame, debug)

            # Inside your video processing loop
            logging.info(f"Processing frame {frame_idx}/{total_frames}")

            
            if detected:
                if not current_attempt:
                    current_attempt.append(frame_idx)
                    clip_id += 1
                    if debug:
                        print(f"New attempt started at frame {frame_idx}, Clip id #{clip_id}, Confidence: {confidence:.2f}")
                else:
                    if debug:
                        print(f"Continuing attempt at frame {frame_idx}, Clip id #{clip_id}, Confidence: {confidence:.2f}")
                current_attempt.append(frame_idx)
            elif current_attempt:
                if (frame_idx - current_attempt[-1]) * skip_frames / fps > detection_drop_threshold:
                    attempts.append((current_attempt[0], current_attempt[-1]))
                    if debug:
                        print(f"Attempt ending at frame {frame_idx} due to detection gap, Clip id #{clip_id}")
                    current_attempt = []
                else:
                    if debug:
                        print(f"Detection gap at frame {frame_idx}, but within threshold, Clip id #{clip_id}")
            
            # Debug: Resize, add frame number, and show processed frame
            if debug:
                display_frame = cv2.resize(frame, None, fx=0.5, fy=0.375, interpolation=cv2.INTER_AREA)
                cv2.putText(display_frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Frame", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break

        frame_idx += 1

    if current_attempt:
        attempts.append((current_attempt[0], current_attempt[-1]))
        if debug:
            print(f"Finalizing last attempt at end of video, Clip id #{clip_id}")

    cap.release()
    cv2.destroyAllWindows()

    for start_frame, end_frame in attempts:
        start_time = max(0, (start_frame / fps) - buffer_before)
        end_time = (end_frame / fps) + buffer_after
        output_path = Path(video_path).stem + f"_clip_{start_time:.2f}_to_{end_time:.2f}.mp4"
        if debug:
            print(f"Generating clip: {output_path}, Start frame: {start_frame}, End frame: {end_frame}")
        generate_clip_with_audio(video_path, start_time, end_time, output_path)



def rider_detected(detections, frame, debug=False):
    highest_confidence = 0
    detected = False
    for *xyxy, conf, cls in detections:
        if int(cls) == 0 and conf > highest_confidence:  # Person class
            highest_confidence = conf
            detected = True
            if debug:
                # Draw bounding box
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    return detected, highest_confidence

def generate_clip_with_audio(video_path, start_time, end_time, output_path):
    # Load the source video to get its total duration
    source_video = VideoFileClip(video_path)
    video_duration = source_video.duration
    
    # Ensure the end time does not exceed the video's duration
    end_time = min(end_time, video_duration)
    
    # Only proceed if the start time is less than the video duration
    if start_time < video_duration:
        video_clip = source_video.subclip(start_time, end_time)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Generated clip with audio: {output_path}")
    else:
        print(f"Skipping generation of {output_path}: start time {start_time} exceeds video duration {video_duration}")
    
    # Close the source video to free resources
    source_video.close()


# Replace 'path/to/your/video.mov' with the actual video file path
process_video('uploaded_videos/IMG_8594.MOV', debug=True)
