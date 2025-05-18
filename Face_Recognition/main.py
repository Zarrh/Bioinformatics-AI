import cv2
from deepface import DeepFace


reference_img = cv2.imread("sus.jpeg")
if reference_img is None:
  print("Reference image not found.")
  exit(1)


MAX_SECONDS = 20


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def compare_local_video(filename="footage1.mp4", outputfile="output.mp4"):

  global MAX_SECONDS

  cap = cv2.VideoCapture(filename)
  if not cap.isOpened():
    print("Could not open video.")
    exit(1)

  fps = cap.get(cv2.CAP_PROP_FPS)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  max_frames = int(fps * MAX_SECONDS)

  out = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

  frame_count = 0
  face_box = None

  print("Starting face verification...")

  while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
      break

    # Show progress
    progress = (frame_count + 1) / max_frames * 100
    print(f"\rProgress: {progress:.2f}%", end="")

    # if frame_count % 30 == 0:
    face_box = None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
      face_crop = frame[y:y+h, x:x+w]
      try:
        result = DeepFace.verify(face_crop, reference_img, enforce_detection=False)
        if result["verified"]:
          face_box = (x, y, w, h)
          break
      except Exception as e:
        print(f"\nVerification error: {e}")

    if face_box:
      x, y, w, h = face_box
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 16, 0), 2)

    out.write(frame)
    frame_count += 1

  cap.release()
  out.release()

  print("\nDone: Video with detection saved.")
  return


if __name__ == "__main__":
  videos = [f"footage{i}.mp4" for i in range(2, 4)]

  for (i, video) in enumerate(videos):
    compare_local_video(filename=video, outputfile=f"output{i+2}.mp4")

  print("All videos saved")
  exit(0)