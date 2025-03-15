import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, height=40, width=512, text_color="black", fg_color="white", font=("Arial", 20))
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
# pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float32  # Use float32 for CPU
)
# pipe.to(device)
pipe.to("cpu")  # Run model on CPU

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]

    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Prevent garbage collection

trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="blue", font=("Arial", 20), command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
