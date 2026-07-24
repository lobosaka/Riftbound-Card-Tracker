import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import datetime

class CameraApp:
    def __init__(self, window, video_source="/dev/video4"):
        self.window = window
        self.window.title("Live Camera Feed")
        
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.vid.set(20, 1)
            
        # Calibration matrices
        self.camera_matrix = np.array(
            [[1239.3367501467612, 0.0, 367.8840428570087], [0.0, 1242.9872838383162, 673.4001708112677], [0.0, 0.0, 1.0]]
        , dtype=np.float32)

        self.dist_coeffs = np.array(
            [[-0.23029730654775918, -0.06403435155721206, 0.00026812586219995214, 0.0002515795799608243, -0.10366338898133773]]
        , dtype=np.float32)
        
        # UI Setup
        self.canvas = tk.Canvas(window, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        ctrl_frame = tk.Frame(window)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Exposure Controls
        self.auto_exposure = True
        self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
        
        self.btn_auto_exp = tk.Button(ctrl_frame, text="Auto Exposure: ON", width=15, command=self.toggle_auto_exposure)
        self.btn_auto_exp.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.scale_exp = tk.Scale(ctrl_frame, from_=100, to=10000, orient=tk.HORIZONTAL, label="Exposure", command=self.set_exposure, state="disabled", resolution=100)
        self.scale_exp.pack(side=tk.LEFT, padx=10, pady=10)
        self.scale_exp.set(166)

        # Rectify Toggle
        self.rectify = True
        self.btn_rectify = tk.Button(ctrl_frame, text="Rectify: ON", width=10, command=self.toggle_rectify)
        self.btn_rectify.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Media Controls
        self.btn_shoot = tk.Button(ctrl_frame, text="Shoot", width=10, command=self.shoot)
        self.btn_shoot.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.btn_record = tk.Button(ctrl_frame, text="Record", width=15, command=self.toggle_record)
        self.btn_record.pack(side=tk.RIGHT, padx=10, pady=10)
        
        self.is_recording = False
        self.out = None
        
        self.update()
        self.window.mainloop()
        
    def process_frame(self, frame):
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rectify:
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return frame

    def toggle_rectify(self):
        self.rectify = not self.rectify
        if self.rectify:
            self.btn_rectify.config(text="Rectify: ON")
        else:
            self.btn_rectify.config(text="Rectify: OFF")

    def toggle_auto_exposure(self):
        self.auto_exposure = not self.auto_exposure
        if self.auto_exposure:
            self.btn_auto_exp.config(text="Auto Exposure: ON")
            self.scale_exp.config(state="disabled")
            self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            self.btn_auto_exp.config(text="Auto Exposure: OFF")
            self.scale_exp.config(state="normal")
            self.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.set_exposure(self.scale_exp.get())

    def set_exposure(self, val):
        if not self.auto_exposure:
            self.vid.set(cv2.CAP_PROP_EXPOSURE, float(val))

    def shoot(self):
        ret, frame = self.vid.read()
        if ret:
            frame = self.process_frame(frame)
            filename = datetime.datetime.now().strftime("photo_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved photo: {filename}")
            
    def toggle_record(self):
        if self.is_recording:
            self.is_recording = False
            self.btn_record.config(text="Record")
            if self.out:
                self.out.release()
                self.out = None
            print("Recording stopped.")
        else:
            ret, frame = self.vid.read()
            if not ret: return
            frame = self.process_frame(frame)
            
            self.is_recording = True
            self.btn_record.config(text="Stop Recording")
            
            filename = datetime.datetime.now().strftime("video_%Y%m%d_%H%M%S.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = self.vid.get(cv2.CAP_PROP_FPS) or 20.0
            
            height, width = frame.shape[:2]
            self.out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            print(f"Recording started: {filename}")

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = self.process_frame(frame)
            
            if self.is_recording and self.out:
                self.out.write(frame)
            
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w > 10 and canvas_h > 10:
                frame_h, frame_w = frame.shape[:2]
                scale = min(canvas_w / frame_w, canvas_h / frame_h)
                new_w, new_h = int(frame_w * scale), int(frame_h * scale)
                
                display_frame = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
                
                x_offset = (canvas_w - new_w) // 2
                y_offset = (canvas_h - new_h) // 2
                
                self.canvas.delete("all")
                self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor=tk.NW)
        
        self.window.after(15, self.update)
        
    def __del__(self):
        if hasattr(self, 'vid') and self.vid.isOpened():
            self.vid.release()
        if hasattr(self, 'out') and self.out:
            self.out.release()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x800")
    app = CameraApp(root)