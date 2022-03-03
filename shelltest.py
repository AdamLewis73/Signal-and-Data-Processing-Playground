from tkinter import *
from tkinter import filedialog
import tkinter

window = Tk()
# file = filedialog.askopenfilename()
filename = ''
window.title('Welcome')
window.geometry('500x300')
# import EMG_Proc_Func.py

def filebrowse():
    filename = filedialog.askopenfilename()
    # win.after()



btn1 = Button(window, text = 'Browse for File (csv only)', command = filebrowse)
btn1.pack(fill=X,padx = 100, pady = 50)



window.mainloop()