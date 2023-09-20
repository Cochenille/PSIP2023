import cv2
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageTk

class RockPaperScissorsApp:
    def __init__(self, root):
        self.root = root
        root.title("Reconnaissance Roche-Papier-Ciseaux")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.sign_to_capture = None
        self.capture_count = 0
        self.image_data = []

        self.create_interface()
        self.create_model()

    def create_interface(self):
        # Créer la vue de la caméra
        self.camera_label = ttk.Label(self.root)
        self.camera_label.pack()

        # Créer des boutons dans l'interface pour la capture
        capture_buttons_frame = ttk.Frame(self.root)
        capture_buttons_frame.pack()

        rock_button = ttk.Button(capture_buttons_frame, text="Roche", command=lambda: self.capture_image("Roche"))
        paper_button = ttk.Button(capture_buttons_frame, text="Papier", command=lambda: self.capture_image("Papier"))
        scissors_button = ttk.Button(capture_buttons_frame, text="Ciseaux", command=lambda: self.capture_image("Ciseaux"))

        rock_button.pack(side="left", padx=10)
        paper_button.pack(side="left", padx=10)
        scissors_button.pack(side="left", padx=10)

        # Afficher le compteur de captures
        self.label_var = tk.StringVar()
        self.label_var.set("Captures : 0")
        label = ttk.Label(self.root, textvariable=self.label_var)
        label.pack()

        # Créer un bouton pour entraîner le modèle
        train_button = ttk.Button(self.root, text="Entraîner le modèle", command=self.train_model)
        train_button.pack(pady=10)

        # Créer un bouton pour jouer
        play_button = ttk.Button(self.root, text="Jouer", command=self.play_game)
        play_button.pack(pady=10)

        # Créer un bouton de réinitialisation
        reset_button = ttk.Button(self.root, text="Réinitialiser", command=self.reset_data)
        reset_button.pack(pady=10)

        # Créer une étiquette pour afficher la prédiction
        self.prediction_label = ttk.Label(self.root, text="")
        self.prediction_label.pack(pady=10)

    def create_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])

    def capture_image(self, sign):
        if self.capture_count < 300:
            self.sign_to_capture = sign
            self.capture_count += 1
            self.label_var.set(f"Captures : {self.capture_count}/300")
        else:
            self.show_info("Information", "Vous avez déjà capturé 300 images.")

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))
        image = image / 255.0  # Normalisation des pixels entre 0 et 1
        return image

    def update_camera_view(self):
        ret, frame = self.cap.read()
        if ret:
            if self.sign_to_capture is not None:
                self.image_data.append([self.preprocess_image(frame), self.sign_to_capture])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.camera_label.img = img
            self.camera_label.config(image=img)
            self.camera_label.after(10, self.update_camera_view)
        else:
            self.show_error("Erreur", "Impossible de lire la caméra.")

    def train_model(self):
        if len(self.image_data) == 0:
            self.show_error("Erreur", "Aucune donnée à entraîner.")
            return

        data = np.array([item[0] for item in self.image_data], dtype=np.float32)
        labels = np.array([item[1] for item in self.image_data])

        labels = np.where(labels == 'Roche', 0, labels)
        labels = np.where(labels == 'Papier', 1, labels)
        labels = np.where(labels == 'Ciseaux', 2, labels)

        labels = tf.keras.utils.to_categorical(labels, 3)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(data, labels, epochs=5, batch_size=32)
        self.show_info("Entraînement terminé", "Le modèle a été entraîné avec succès.")

    def play_game(self):
        ret, frame = self.cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
            img = self.preprocess_image(img)
            img = np.expand_dims(img, axis=0)
            prediction = self.model.predict(img)
            sign = ["Roche", "Papier", "Ciseaux"][np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            self.prediction_label.config(text=f"Prédiction : {sign}\nConfiance : {confidence:.2f}%")
        else:
            self.show_error("Erreur", "Impossible de lire la caméra.")

    def reset_data(self):
        self.sign_to_capture = None
        self.capture_count = 0
        self.image_data = []
        self.label_var.set("Captures : 0")
        self.prediction_label.config(text="")


    def run(self):
        self.update_camera_view()
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = RockPaperScissorsApp(root)
    app.run()
