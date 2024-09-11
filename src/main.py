import cv2
import time


def take_picture_with_delay(filename="captured_image.jpg", delay=5):
    # Open the default camera (usually the webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    print(f"Get ready! Taking picture in {delay} seconds...")

    start_time = time.time()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not capture frame.")
            break

        # Display the live video feed
        cv2.imshow("Live Preview - Smile!", frame)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Break the loop after the specified delay
        if elapsed_time > delay:
            break

        # If the user presses 'q', exit early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User exited before capture.")
            break

    # Save the last frame after the delay
    cv2.imwrite(filename, frame)
    print(f"Image saved as {filename}")

    # Release the camera and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    take_picture_with_delay()
