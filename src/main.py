import argparse
import cv2
import time
import torch
from torchvision import transforms
import os
import CNN_lib


def crop_to_square(frame):
    height, width, _ = frame.shape

    # Izracunavamo centar da bismo posle uradili crop/scale
    min_dim = min(height, width)
    top = (height - min_dim) // 2
    left = (width - min_dim) // 2

    cropped_frame = frame[top : top + min_dim, left : left + min_dim]

    return cropped_frame


def take_picture_and_resize(filename, delay=5, target_size=(48, 48)):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Nije moguce otvoriti video capture!")
        return

    print(f"Priprema! Slikanje za {delay} sekundi...")

    start_time = time.time()

    while True:
        # Citanje jednog frejma
        ret, frame = cap.read()

        if not ret:
            print("Error: Neuspesno citanje frejma.")
            break

        # Prikazujemo uzivo feed
        cv2.imshow("Live Preview!", frame)
        elapsed_time = time.time() - start_time
        if elapsed_time > delay:
            break

        # Izlazenje pre zavrsetka
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Korisnik je prekinuo slikanje.")
            break

    square_frame = crop_to_square(frame)
    cv2.imwrite(filename + ".jpg", square_frame)
    resized_frame = cv2.resize(square_frame, target_size)

    cv2.imwrite("face.jpg", resized_frame)
    print(f"Slika je sacuvana: {filename}.")

    cap.release()
    cv2.destroyAllWindows()


def load_image_as_tensor_opencv(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    image_tensor = transform(image)

    # Ovo je potrebno jer nas model ocekuje shape (1,1,48,48) jer smo radili sa batch-ovima
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Face expression and emotion recognition of the user."
    )

    parser.add_argument(
        "--save_name",
        type=str,
        required=False,
        help="Optional name for saving the picture",
        default="captured_image.jpg",
    )

    args = parser.parse_args()

    take_picture_and_resize(filename=args.save_name)

    tensor_image = load_image_as_tensor_opencv("./face.jpg")
    emotions = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprise",
    }

    # Ucitamo model koji cemo da koristimo
    # print(os.path.curdir)
    model = CNN_lib.EmotionNet()
    model.load_state_dict(torch.load("./models/emoNet.pth", weights_only=True))
    print("Model ucitan, prepoznavanje facijalnih eksrpresija...")
    model.eval()

    with torch.no_grad():
        output = model(tensor_image)
        print(output)
        _, predicted_class = torch.max(output, 1)
        print(f"Predicted class: {emotions[predicted_class.item()]}")
