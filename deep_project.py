import cv2
import torch
import numpy as np
from PIL import Image


# 1. Device and model loading

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

print("Loading AnimeGANv2 model from torch.hub ...")
generator = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=device,
).eval()

face2paint = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "face2paint",
    size=512,
    device=device,
)
print("Model loaded.")



# 2. Helper: frame -> anime

@torch.no_grad()
def animefy_frame_bgr(frame_bgr):
    """
    Input:  BGR frame from OpenCV.
    Output: BGR frame with AnimeGANv2 style.
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # To PIL
    img_pil = Image.fromarray(img_rgb).convert("RGB")
    # AnimeGANv2
    out_pil = face2paint(generator, img_pil)
    # Back to OpenCV BGR
    out_rgb = np.array(out_pil)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


# 3. Webcam loop

def main():
    # 0 = default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

      
        frame = cv2.flip(frame, 1)

        
        anime_frame = animefy_frame_bgr(frame)

        # Show both
        cv2.imshow("Webcam - Original", frame)
        cv2.imshow("Webcam - Anime", anime_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

