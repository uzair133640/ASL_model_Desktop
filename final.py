# import tkinter as tk
# from tkinter import Label, Button
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf  # Add at the top
# from tensorflow.keras.models import load_model
# from PIL import Image, ImageTk
# import time
# from collections import deque

# CATEGORIES = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

# class ASLRecognizer:
#     def __init__(self, window):
#         self.window = window
#         self.window.title("ASL Real-Time Recognition")
        
#         # Video Capture
#         self.cap = cv2.VideoCapture(0)
#         self.frame_queue = deque(maxlen=1)
        
#         # Mediapipe
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7
#         )
#         self.mp_drawing = mp.solutions.drawing_utils
        
#         # Model
#         self.model = tf.keras.models.load_model("asl_landmark_model.h5")

#         # GUI Elements
#         self.video_label = Label(self.window)
#         self.video_label.pack()

#         self.current_char = ""
#         self.last_prediction = ""
#         self.last_append_time = time.time()
#         self.prediction_history = deque(maxlen=5)

#         self.output_label = Label(self.window, text="", font=("Helvetica", 32))
#         self.output_label.pack()

#         self.sentence = ""
#         self.sentence_label = Label(self.window, text="", font=("Helvetica", 24))
#         self.sentence_label.pack()

#         self.add_button = Button(self.window, text="Add Character", command=self.add_character)
#         self.add_button.pack()

#         self.space_button = Button(self.window, text="Add Space", command=self.add_space)
#         self.space_button.pack()

#         self.clear_button = Button(self.window, text="Clear", command=self.clear_sentence)
#         self.clear_button.pack()

#         self.running = True
#         self.processing = False

#         self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
#         self.update_video()
#         self.window.after(10, self.process_loop)

#     def normalize_landmarks(self, landmarks):
#         landmarks = np.array(landmarks).reshape(-1, 3)
#         wrist = landmarks[0]
#         landmarks -= wrist
#         max_val = np.max(np.abs(landmarks))
#         return landmarks.flatten() / max_val if max_val > 0 else landmarks.flatten()

#     def extract_features(self, results):
#         features = []
#         hand_count = len(results.multi_hand_landmarks)
        
#         for i in range(2):  # Ensure 2-hand input
#             if i < hand_count:
#                 lm = results.multi_hand_landmarks[i]
#                 handedness = results.multi_handedness[i].classification[0].label
#                 lm_list = [[p.x, p.y, p.z] for p in lm.landmark]
#                 norm_lm = self.normalize_landmarks(lm_list)
#                 features.extend(norm_lm)
#                 features.append(1.0 if handedness == "Right" else 0.0)
#             else:
#                 features.extend([0.0] * 63)  # Empty hand
#                 features.append(0.0)
#         return np.array(features).reshape(1, -1)

#     def update_video(self):
#         if self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 frame = cv2.flip(frame, 1)
#                 self.frame_queue.append(frame.copy())
#             self.window.after(10, self.update_video)

#     def process_loop(self):
#         if self.running and self.frame_queue and not self.processing:
#             self.processing = True
#             frame = self.frame_queue[-1]
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = self.hands.process(rgb)

#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(
#                         rgb, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                     )
#                 features = self.extract_features(results)
#                 pred = self.model.predict(features, verbose=0)[0]
#                 conf = np.max(pred)
#                 char = CATEGORIES[np.argmax(pred)]

#                 if conf > 0.8:
#                     self.prediction_history.append(char)
#                     # Majority vote for stability
#                     stable_char = max(set(self.prediction_history), key=self.prediction_history.count)
#                     self.current_char = stable_char

#                     # Auto-add after 1.5s if new char
#                     if stable_char != self.last_prediction and time.time() - self.last_append_time > 1.5:
#                         self.add_character()
#                         self.last_append_time = time.time()
#                         self.last_prediction = stable_char
#                 else:
#                     self.current_char = ""
#             else:
#                 self.current_char = ""

#             imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
#             self.video_label.imgtk = imgtk
#             self.video_label.config(image=imgtk)

#             self.output_label.config(text=self.current_char)
#             self.sentence_label.config(text=self.sentence)
#             self.processing = False

#         self.window.after(10, self.process_loop)

#     def add_character(self):
#         if self.current_char:
#             self.sentence += self.current_char
#             self.sentence_label.config(text=self.sentence)

#     def add_space(self):
#         self.sentence += " "
#         self.sentence_label.config(text=self.sentence)

#     def clear_sentence(self):
#         self.sentence = ""
#         self.sentence_label.config(text=self.sentence)
#         self.last_prediction = ""
#         self.prediction_history.clear()

#     def on_closing(self):
#         self.running = False
#         self.cap.release()
#         self.window.destroy()

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = ASLRecognizer(root)
#     root.mainloop()










import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
import pyttsx3
import threading
import queue
import time
import difflib

# Constants
CATEGORIES = [
    *list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    'del', 'nothing', 'space'
]
# A larger dictionary you can expand as desired
WORD_DICT = ["WORK", "WORLD", "HELLO", "MORE", "HELP", "HOUSE", "HOW", "ARE", "YOU", "WHAT", "IS", "THIS"]

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language To Text Conversion")
        self.root.geometry("1000x700")

        # Load model + TTS
        self.model = tf.keras.models.load_model('asl_model_mediapipe.h5')
        self.engine = pyttsx3.init()

        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # State
        self.sentence = ""
        self.current_char = ""
        self.last_prediction = ""
        self.last_append_time = time.time()
        self.frame_queue = queue.Queue(maxsize=1)
        self.processing = False
        self.running = True

        # Build GUI & start video
        self.setup_gui()
        self.start_camera_threads()
        self.update_gui_loop()

    def setup_gui(self):
        top = tk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video
        self.video_label = tk.Label(top)
        self.video_label.pack(side=tk.LEFT, padx=10)

        # Current character
        right = tk.Frame(top)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(right, text="Character", font=("Arial", 14)).pack(pady=(0,10))
        self.char_display = tk.Label(right, text="", font=("Arial", 120), bg="#f0f0f0")
        self.char_display.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Sentence
        sentence_fr = tk.Frame(self.root)
        sentence_fr.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(sentence_fr, text="Sentence", font=("Arial", 14)).pack(anchor=tk.W)
        self.sentence_var = tk.StringVar()
        self.sentence_entry = tk.Entry(
            sentence_fr, textvariable=self.sentence_var,
            font=("Arial", 24), state='readonly', readonlybackground='white'
        )
        self.sentence_entry.pack(fill=tk.X, pady=5)

        # Buttons
        btn_fr = tk.Frame(self.root)
        btn_fr.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(btn_fr, text="Clear All", command=self.clear_sentence, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_fr, text="Clear Last", command=self.clear_last_char, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_fr, text="Speak", command=self.speak_sentence, width=12).pack(side=tk.LEFT, padx=5)

        # Suggestions
        sug_fr = tk.Frame(self.root)
        sug_fr.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(sug_fr, text="Suggestions", font=("Arial", 14)).pack(anchor=tk.W)
        self.suggestion_btns = []
        for _ in range(4):
            b = tk.Button(sug_fr, text="", width=10, command=lambda: None)
            b.pack(side=tk.LEFT, padx=5)
            self.suggestion_btns.append(b)

    def start_camera_threads(self):
        # Open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            self.root.destroy()
            return

        # Start threads
        threading.Thread(target=self.capture_loop, daemon=True).start()
        threading.Thread(target=self.process_loop, daemon=True).start()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_loop(self):
        while self.running:
            if not self.frame_queue.empty() and not self.processing:
                self.processing = True
                frame = self.frame_queue.get()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                # Draw landmarks & predict
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        rgb, lm, self.mp_hands.HAND_CONNECTIONS
                    )
                    data = [c for p in lm.landmark for c in (p.x, p.y, p.z)]
                    pred = self.model.predict(
                        np.array(data).reshape(1,-1), verbose=0
                    )[0]
                    conf = np.max(pred)
                    char = CATEGORIES[np.argmax(pred)]
                    if conf > 0.8:
                        if char != self.last_prediction:
                            self.current_char = char
                            self.last_prediction = char

                        # Auto-append every 1.5s
                        if time.time() - self.last_append_time > 1.5:
                            self.add_character()
                            self.last_append_time = time.time()
                    else:
                        self.current_char = ""

                # Update video frame
                imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                self.processing = False

    def update_gui_loop(self):
        # Char display
        disp = self.current_char if self.current_char not in ('nothing','del','space') else ""
        self.char_display.config(text=disp)
        # Sentence display
        self.sentence_var.set(self.sentence)
        # Next iteration
        self.root.after(100, self.update_gui_loop)

    def add_character(self):
        c = self.current_char
        if not c:
            return
        if c == 'space':
            self.sentence += ' '
        elif c == 'del':
            self.sentence = self.sentence[:-1]
        elif c not in ('nothing', 'del', 'space'):
            self.sentence += c
        self.update_suggestions()

    def update_suggestions(self):
        last = self.sentence.strip().split(" ")[-1].upper()
        # get up to 4 prefix matches, fallback to close matches
        matches = [w for w in WORD_DICT if w.startswith(last)] \
                  or difflib.get_close_matches(last, WORD_DICT, n=4, cutoff=0.5)
        for i, btn in enumerate(self.suggestion_btns):
            if i < len(matches):
                btn.config(
                    text=matches[i],
                    command=lambda w=matches[i]: self.apply_suggestion(w)
                )
            else:
                btn.config(text="", command=lambda: None)

    def apply_suggestion(self, word):
        parts = self.sentence.strip().split(" ")
        parts[-1] = word
        self.sentence = " ".join(parts) + " "
        self.update_suggestions()

    def clear_sentence(self):
        self.sentence = ""
        self.update_suggestions()

    def clear_last_char(self):
        self.sentence = self.sentence[:-1]
        self.update_suggestions()

    def speak_sentence(self):
        threading.Thread(
            target=lambda: (self.engine.say(self.sentence), self.engine.runAndWait()),
            daemon=True
        ).start()

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
