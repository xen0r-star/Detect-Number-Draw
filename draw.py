from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from Network.ModeleDetect import ModeleDetect

first_color = "#C6C6C6"
epaisseur = 20
modele = "Network/Model/model2.pkl"

win = Tk()
win.title("Detect Number Draw")
win.iconbitmap("Picture/Logo.ico")
win.geometry("500x600")
win.resizable(False, False)
win.configure(bg=first_color)
win.grid_columnconfigure(0, weight=1)


picture_Logo = ImageTk.PhotoImage(image=Image.open("Picture/Logo 2.png").resize((374, 40), Image.LANCZOS))
Logo = Label(win, width=374, height=40, image=picture_Logo, bg=first_color, bd=0, highlightthickness=0, activebackground=first_color)
Logo.grid(row=0, columnspan=3, pady=20)

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
    
    picture_Result = ImageTk.PhotoImage(image=Image.open("Picture/Result.png").resize((40, 40), Image.LANCZOS))
    Result.config(image=picture_Result)
    Result.image = picture_Result
    

Canvas = Canvas(win, width=420, height=420, bg="#ffffff", highlightthickness=3, highlightbackground="#000000")
Canvas.grid(row=1, columnspan=3, pady=(0,20))
Canvas.create_rectangle(0, 0, 420, 420, fill="#ffffff", width=0)

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


Draw = Frame(win, bg=first_color)
Draw.grid(row=2, column=0, sticky=W, padx=(40, 0))

def draw_statut(var):
    global draw_color_stat

    if var == 1:
        draw_color_stat = "#000000"
        picture_Draw_black_on = ImageTk.PhotoImage(image=Image.open("Picture/Draw black on.png").resize((40, 40), Image.LANCZOS))
        Draw_black.config(image=picture_Draw_black_on)
        Draw_black.image = picture_Draw_black_on

        picture_Draw_white_on = ImageTk.PhotoImage(image=Image.open("Picture/Draw white.png").resize((40, 40), Image.LANCZOS))
        Draw_white.config(image=picture_Draw_white_on)
        Draw_white.image = picture_Draw_white_on
    else:
        draw_color_stat = "#ffffff"
        picture_Draw_white_on = ImageTk.PhotoImage(image=Image.open("Picture/Draw white on.png").resize((40, 40), Image.LANCZOS))
        Draw_white.config(image=picture_Draw_white_on)
        Draw_white.image = picture_Draw_white_on

        picture_Draw_black_on = ImageTk.PhotoImage(image=Image.open("Picture/Draw black.png").resize((40, 40), Image.LANCZOS))
        Draw_black.config(image=picture_Draw_black_on)
        Draw_black.image = picture_Draw_black_on

picture_Draw_black = ImageTk.PhotoImage(image=Image.open("Picture/Draw black on.png").resize((40, 40), Image.LANCZOS))
Draw_black = Button(Draw, command=lambda: draw_statut(1), image=picture_Draw_black, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
Draw_black.grid(row=0, column=0, ipadx=2, ipady=2)

picture_Draw_white = ImageTk.PhotoImage(image=Image.open("Picture/Draw white.png").resize((40, 40), Image.LANCZOS))
Draw_white = Button(Draw, command=lambda: draw_statut(2), image=picture_Draw_white, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
Draw_white.grid(row=0, column=1, ipadx=2, ipady=2, padx=5)

picture_Bucket = ImageTk.PhotoImage(image=Image.open("Picture/Bucket.png").resize((40, 40), Image.LANCZOS))
Bucket = Button(Draw, command=Canvas_delete, image=picture_Bucket, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
Bucket.grid(row=0, column=2, ipadx=2, ipady=2)

def Detect_number():
    picture_Result = ImageTk.PhotoImage(image=Image.open("Picture/Result load.png").resize((40, 40), Image.LANCZOS))
    Result.config(image=picture_Result)
    Result.image = picture_Result

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
    # color_px_line = color_px_short.flatten()
    color_px_line = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    answer = ModeleDetect(color_px_line, modele)

    if answer >= 0 and answer <= 9:
        picture_Result = ImageTk.PhotoImage(image=Image.open("Picture/Result " + str(answer) + ".png").resize((40, 40), Image.LANCZOS))
        Result.config(image=picture_Result)
        Result.image = picture_Result
    else:
        picture_Result = ImageTk.PhotoImage(image=Image.open("Picture/Result load.png").resize((40, 40), Image.LANCZOS))
        Result.config(image=picture_Result)
        Result.image = picture_Result


picture_Detected = ImageTk.PhotoImage(image=Image.open("Picture/Detected.png").resize((160, 40), Image.LANCZOS))
Detected = Button(win, command=lambda: Detect_number(), image=picture_Detected, bg=first_color, cursor="hand2", bd=0, highlightthickness=0, activebackground=first_color)
Detected.grid(row=2, column=1, ipadx=2, ipady=2, sticky=NSEW, padx=36)


picture_Result = ImageTk.PhotoImage(image=Image.open("Picture/Result.png").resize((40, 40), Image.LANCZOS))
Result = Label(win, image=picture_Result, bg=first_color, bd=0, highlightthickness=0, activebackground=first_color)
Result.grid(row=2, column=2, sticky=E, padx=(0, 40))

Canvas_delete()

win.mainloop()
