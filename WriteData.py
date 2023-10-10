from os import path
import shutil
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

first_color = "#C6C6C6"
epaisseur = 20
file = "Data.dat"

win = Tk()
win.title("Detect Number Draw - WriteData")
win.iconbitmap("Picture/Logo.ico")
win.geometry("700x600")
win.resizable(False, False)
win.configure(bg=first_color)
win.grid_columnconfigure(0, weight=1)


picture_Logo = ImageTk.PhotoImage(image=Image.open("Picture/Logo 2.png").resize((374, 40), Image.LANCZOS))
Logo = Label(win, width=374, height=40, image=picture_Logo, bg=first_color, bd=0, highlightthickness=0, activebackground=first_color)
Logo.grid(row=0, columnspan=13, pady=20)

color_px = {}
color_px_short = {}
color_px_line = {}
def Canvas_delete():
    global color_px
    global color_px_short
    global color_px_line

    Canvas.delete(ALL)
    Canvas.create_rectangle(0, 0, 420, 420, fill="#ffffff", width=0)
    color_px = np.zeros((420, 420), dtype=int)
    color_px_short = np.zeros((28, 28), dtype=int)
    color_px_line = [0] * (784)
    

Canvas = Canvas(win, width=420, height=420, bg="#ffffff", highlightthickness=3, highlightbackground="#000000")
Canvas.grid(row=1, columnspan=13, pady=(0,20))
Canvas.create_rectangle(0, 0, 420, 420, fill="#ffffff", width=0)
Canvas_delete()

etatboutonsouris='haut'
draw_color_stat = "#000000"

def clic(event):
    global etatboutonsouris,x1,y1
    etatboutonsouris='bas'      
    x1=event.x
    y1=event.y
Canvas.bind('<ButtonPress-1>', clic)
 
def mouvement(event):
    global etatboutonsouris,x,y
    if (etatboutonsouris=='bas' and (x <= 420 and x >= 0) and (y <= 420 and y >= 0)):
        Canvas.create_rectangle(x, y, event.x, event.y, outline=draw_color_stat, fill=draw_color_stat, width=epaisseur)
    
    x=event.x
    y=event.y
Canvas.bind('<Motion>',mouvement)

def declic(event):
    global etatboutonsouris,x2,y2
    etatboutonsouris='haut'
    x2=event.x
    y2=event.y
Canvas.bind('<ButtonRelease-1>', declic)

def Detect_number(category):
    global color_px
    global color_px_short
    global color_px_line
    for a in range(420):
        for b in range(420):
            pixel = Canvas.find_closest(a, b)
            color = Canvas.itemcget(pixel, "fill")

            if color == "#000000":
                c = 1
            else: 
                c = 0
            color_px[a, b] = c
    
    for i in range(28):
        for j in range(28):
            moyenne = np.mean(color_px[i*15:(i+1)*15, j*15:(j+1)*15]) # Calcul de la moyenne dans le carré 28x28
            color_px_short[i, j] = 1 if moyenne > 0.5 else 0

    color_px_short = np.rot90(np.flipud(color_px_short), k=-1) # rotation de -90° et symétrie orthogonale 
    color_px_line = color_px_short.flatten()
    color_px_list = color_px_line.tolist()

    ReadData(file, category, color_px_list)
    Canvas_delete()

def ReadData(filename, category, value):
    if not path.isfile(filename):
        raise FileNotFoundError("File not found")

    with open(filename, 'r') as fp:
        data = fp.read()

    data_dict = {}
    if data:
        lines = data.strip().split('\n')
        for line in lines:
            key, val = line.split(': ')
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(val)

    if category not in data_dict:
        data_dict[category] = []
    data_dict[category].append(value)

    new_data = '\n'.join([f"{key}: {val}" for key, values in data_dict.items() for val in values])

    with open(filename, 'w') as fp:
        fp.write(new_data)

def Save():
    if not path.isfile(file):
        raise FileNotFoundError("Source file not found")

    user_downloads_dir = str(Path.home() / "Downloads")
    filename = path.basename(file)
    destination_file = path.join(user_downloads_dir, filename)
    shutil.copy2(file, destination_file)

def Send():
    file_path = filedialog.askopenfilename(
        initialdir=str(Path.home() / "Downloads"),
        filetypes=[("Fichiers DAT", "*.dat")]
    )

    if file_path:
        shutil.copy2(file_path, "Data.dat")


Draw = Frame(win, bg=first_color)
Draw.grid(row=2, column=0)

picture_Bucket = ImageTk.PhotoImage(image=Image.open("Picture/Bucket.png").resize((40, 40), Image.LANCZOS))
Bucket = Button(Draw, command=Canvas_delete, image=picture_Bucket, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
Bucket.grid(row=0, column=0, ipadx=2, ipady=2, padx=(0, 20))

for i in range(10):
    image = Image.open(f"Picture/Result {i}.png").resize((40, 40), Image.LANCZOS)
    picture_number = ImageTk.PhotoImage(image=image)
    number = Button(Draw, command=lambda i=i: Detect_number(i), image=picture_number, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
    number.grid(row=0, column=i + 1, ipadx=2, ipady=2)
    number.image = picture_number

picture_save = ImageTk.PhotoImage(image=Image.open("Picture/Save.png").resize((40, 40), Image.LANCZOS))
save = Button(Draw, command=Save, image=picture_save, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
save.grid(row=0, column=11, ipadx=2, ipady=2, padx=(20,0))

picture_send = ImageTk.PhotoImage(image=Image.open("Picture/Send.png").resize((40, 40), Image.LANCZOS))
send = Button(Draw, command=Send, image=picture_send, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
send.grid(row=0, column=12, ipadx=2, ipady=2)


win.mainloop()