from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2


def main():
    print("Starting the program!")

    model = YOLO("yolo11n-pose.pt")

    total_count = 0

    # The line
    LINE_Y = 300
    line_start = (0, LINE_Y)
    line_end = (640, LINE_Y)

    # Store the last known side: 0 for above, 1 for below
    l_last_side = None 
    r_last_side = None

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Draw the counting line
        cv2.line(annotated_frame, line_start, line_end, (0, 0, 255), 5)

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            points = results[0].keypoints.xy[0]

            if len(points) > 10:
                # --- Right hand logic ---
                rx, ry = int(points[10][0].item()), int(points[10][1].item())
                if rx > 0 and ry > 0:
                    cv2.circle(annotated_frame, (rx, ry), 15, (255, 0, 0), -1)
                    
                    # Determine current side (0 or 1)
                    current_r_side = 1 if ry > LINE_Y else 0
                    
                    # If this isn't the first frame and the side changed...
                    if r_last_side is not None and current_r_side != r_last_side:
                        total_count += 1
                        print(f"PASS DETECTED! Total: {total_count}")
                    
                    r_last_side = current_r_side

                # --- Left hand logic ---
                lx, ly = int(points[9][0].item()), int(points[9][1].item())
                if lx > 0 and ly > 0:
                    cv2.circle(annotated_frame, (lx, ly), 15, (255, 0, 0), -1)
                    
                    current_l_side = 1 if ly > LINE_Y else 0
                    
                    if l_last_side is not None and current_l_side != l_last_side:
                        total_count += 1
                        print(f"PASS DETECTED! Total: {total_count}")
                    
                    l_last_side = current_l_side

        cv2.putText(annotated_frame, f"Count: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Basic Pose Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
