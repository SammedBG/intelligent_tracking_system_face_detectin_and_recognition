import os
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Training Capture Interface")

        self.label = tk.Label(root, text="Enter Name:")
        self.label.pack()

        self.name_entry = tk.Entry(root)
        self.name_entry.pack()

        self.capture_btn = tk.Button(root, text="Start Capture", command=self.start_capture)
        self.capture_btn.pack()

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video)

    def start_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        save_path = os.path.join("data", "employee_images", name)
        os.makedirs(save_path, exist_ok=True)

        count = 0
        max_images = 40
        while count < max_images:
            ret, frame = self.cap.read()
            if not ret:
                break
            face_path = os.path.join(save_path, f"{name}_{count+1}.jpg")
            cv2.imwrite(face_path, frame)
            count += 1
            cv2.imshow("Capturing Faces", frame)
            cv2.waitKey(200)  # wait 200 ms between captures

        cv2.destroyWindow("Capturing Faces")
        messagebox.showinfo("Done", f"Captured {max_images} images for {name}")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()
