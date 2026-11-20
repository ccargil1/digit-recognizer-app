from neural_network.layers import Dense
from neural_network.network import NeuralNetwork
from PIL import Image, ImageDraw
import tkinter as tk
import customtkinter as ctk
import numpy as np

root = ctk.CTk()
root.title("Handwritten Digit Recognizer")
root.geometry("600x600")
root.configure(fg_color="#181818")

image = Image.new(
    mode="L", 
    size=(280, 280), 
    color=0
)
image_draw = ImageDraw.Draw(image)

network = NeuralNetwork()
network.add_layer(Dense(784, 128))
network.add_layer(Dense(128, 64))
network.add_layer(Dense(64, 10))

previous_position = None

def load_model():
    with np.load("models/best_model.npz") as saved_data:
        for i, layer in enumerate(network.layers):
            layer.weights = saved_data[f"weights_{i}"]
            layer.biases = saved_data[f"biases_{i}"]

def get_bounding_box(x: int, y: int, radius: int) -> tuple[int, int, int, int]:
    return (x - radius, y - radius, x + radius, y + radius)

def draw(event: tk.Event):
    global previous_position
    
    x = event.x
    y = event.y
    
    if not previous_position is None:
        x1, y1, x2, y2 = get_bounding_box(event.x, event.y, 10)
        
        canvas.create_line(
            previous_position[0], previous_position[1],
            x, y,
            fill="white",
            width=20,
            capstyle=tk.ROUND
        )
        
        image_draw.line(
            [previous_position, (x, y)],
            fill=255,
            width=20
        )
        
        image_draw.ellipse(
            (x1, y1, x2, y2),
            fill=255
        )
    
    previous_position = (x, y)
    
def start_drawing(event: tk.Event):
    global previous_position
    previous_position = (event.x, event.y)
    
    predict_btn.configure(state="normal")
    
    canvas.create_oval(
        get_bounding_box(event.x, event.y, 10),
        fill="white",
        outline="white"
    )
    
    image_draw.ellipse(
        get_bounding_box(event.x, event.y, 10),
        fill=255
    )
    
def stop_drawing(event: tk.Event):
    global previous_position
    previous_position = None
    
def clear_canvas():
    global previous_position
    previous_position = None
    
    predict_btn.configure(state="disabled")
    
    canvas.delete("all")
    image.paste(0, (0, 0, 280, 280))
    
    predicted_digit_lbl.configure(text="-")
    confidence_lbl.configure(text="--.-% confidence")

def predict_digit():
    resized_image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )
    
    pixels = np.asarray(resized_image, dtype=np.float64) / 255
    pixels = pixels.reshape(784, 1)
    
    probabilities = network.forward(pixels)
    prediction = np.argmax(probabilities)
    confidence = np.max(probabilities)
    
    predicted_digit_lbl.configure(text=str(prediction))
    confidence_lbl.configure(
        text=f"{confidence * 100:.1f}% confidence"
    )
    
load_model()

instructions_lbl = ctk.CTkLabel(
    root,
    text="Draw a digit (0-9)",
    text_color="#aeaeae",
    height=10,
    font=ctk.CTkFont(size=16)
)
instructions_lbl.pack(pady=(15,15))

canvas = ctk.CTkCanvas(
    root,
    width=280,
    height=280,
    bg="#111111",
    highlightthickness=0,
)
canvas.pack(pady=(0,15))

predicted_lbl = ctk.CTkLabel(
    root,
    text="PREDICTED DIGIT",
    text_color="#aeaeae",
    height=10,
    font=ctk.CTkFont(size=16)
)
predicted_lbl.pack(pady=(0,15))

predicted_digit_lbl = ctk.CTkLabel(
    root,
    text="-",
    text_color="#FFFFFF",
    height=70,
    font=ctk.CTkFont(size=70)
)
predicted_digit_lbl.pack()

confidence_lbl = ctk.CTkLabel(
    root,
    text="--.-% confidence",
    text_color="#FFFFFF",
    height=50,
    font=ctk.CTkFont(size=16)
)
confidence_lbl.pack(pady=(0,15))
    
button_frm = ctk.CTkFrame(
    root,
    fg_color="transparent",
)
button_frm.pack(pady=0)
    
clear_btn = ctk.CTkButton(
    button_frm,
    text="Clear",
    fg_color="#141414",
    text_color="#FFFFFF",
    width=175,
    height=45,
    font=ctk.CTkFont(size=18),
    cursor="hand2",
    corner_radius=12,
    command=clear_canvas,
)
clear_btn.pack(side="left", padx=5)

predict_btn = ctk.CTkButton(
    button_frm,
    text="Predict",
    fg_color="#141414",
    text_color="#FFFFFF",
    width=175,
    height=45,
    font=ctk.CTkFont(size=18),
    cursor="hand2",
    corner_radius=12,
    command=predict_digit,
    state="disabled",
)
predict_btn.pack(side="left", padx=5)

canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)

root.mainloop()