import os, time, cv2,datetime,traceback,sys
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import style
style.use('seaborn-darkgrid')
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import pickle




class Make_Contours(tk.Frame):
    def __init__(self, parent, dirs=['Input',],extension='.tif'):
        tk.Frame.__init__(self)
        #Initialize attributes
        self.parent = parent
        parent.title('Contour Extraction')
        self.dirs = dirs
        self.extension = extension
        self.drawmode = None
        self.coords_count = -1
        self.coords = []
        self.parent.bind('<Key>',self.key_bindings)
        s=ttk.Style()
##        print(s.theme_names())
        s.theme_use('xpnative')
        self.frame09=tk.Frame(master=parent)
        self.frame09.grid(column=0,row=0)
        self.frame1=tk.Frame(master=parent)
        self.frame1.grid(column=0,row=3)
        self.frame2=tk.Frame(master=parent)
        self.frame2.grid(column=0,row=4)
        self.frame3=tk.Frame(master=parent)
        self.frame3.grid(column=0,row=5)
        self.frame4=tk.Frame(master=parent)
        self.frame4.grid(column=0,row=6)

        menubar = tk.Menu(self.frame09)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_file)
        filemenu.add_command(label="Save", command=self.save_contour)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        parent.config(menu=menubar)

        drawmenu = tk.Menu(menubar, tearoff=0)
        drawmenu.add_command(label="Create Closed Contour", command=self.black_pixel)
        drawmenu.add_command(label="Pause (shortcut='p')", command=self.exit_drawing)
        drawmenu.add_separator()
        drawmenu.add_command(label='Next Image (Right)',command=self.next_image)
        drawmenu.add_command(label='Previous Image (Left)',command=self.prev_image)
        drawmenu.add_command(label="Undo (shortcut=z)", command=self.undo_draw)
        drawmenu.add_command(label="Redo (shortcut=z)", command=self.redo_draw)
        drawmenu.add_command(label='Next Countour (Up)',command=self.next_con)
        drawmenu.add_command(label='Previous Countour (Down)',command=self.prev_con)
        menubar.add_cascade(label="Draw", menu=drawmenu)
        parent.config(menu=menubar)
        
        self.fig,self.ax = plt.subplots(1,1)
        #plt.gca().invert_yaxis()
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.03, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        self.xmax,self.ymax=1200,1600
        self.im = 255 * np.ones(shape=[self.xmax,self.ymax, 3], dtype=np.uint8)
        self.image = self.ax.imshow(self.im)
        #self.fig.set_size_inches((8,8))
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame1)
        self.toolbar.update()
        self.canvas._tkcanvas.config(bg='blue')
        self.canvas._tkcanvas.pack()
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.callbacks.connect('button_press_event', self.callback)



        tk.Label(self.frame2,text='Input Parameters: ').grid(column=0,row=1)
        self.e1,self.e2,self.e3 = tk.Entry(self.frame2, width=9),tk.Entry(self.frame2,width=9),tk.Entry(self.frame2,width=9)
        self.e1.grid(row=1,column=2)
        self.e2.grid(row=1,column=4)
        self.e3.grid(row=1,column=6)
        tk.Label(self.frame2,text='Name').grid(column=1,row=1)
        tk.Label(self.frame2,text='Material').grid(column=3,row=1)
        tk.Label(self.frame2,text='Curing Time').grid(column=5,row=1)
        self.params = dict(name=self.e1,material=self.e2,Print_Time=self.e3)

        
        self.prev = tk.Button(self.frame4,text='Previous',command=self.prev_image)
        self.prev.grid(row=0,column=3)
        self.next = tk.Button(self.frame4,text='Next',command=self.next_image)
        self.next.grid(row=0,column=4)

        blackbutton = tk.Button(self.frame4,text='New Contour',command=self.black_pixel)
        blackbutton.grid(column=3,row=1)
##        undobutton = tk.Button(self.frame4,text='Undo',command=self.undo_draw)
##        undobutton.grid(column=4,row=1)
##        drawingexit = tk.Button(self.frame4,text='Pause',command=self.exit_drawing)
##        drawingexit.grid(column=5,row=1)
        savedrawing = tk.Button(self.frame4,text='Save',command=self.save_contour)
        savedrawing.grid(column=6,row=1)
        tk.Button(self.frame4,text='Clear',command=self.clear).grid(column=7,row=1)
##        tk.Button(self.frame4,text='Next Countour',command=self.next_con).grid(column=8,row=1)
##        tk.Button(self.frame4,text='Previous Contour',command=self.prev_con).grid(column=9,row=1)
        self.coords = []
        self.redo = []
        self.make_list()
        self.open_image()

    def open_file(self):
        from tkinter import filedialog
        filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("pickle files","*.pickle"),("all files","*.*")))
        print(filename)
        with open(filename,"rb") as file:
            pickle_object = pickle.load(file)
        self.im = pickle_object['Original_image']
        self.coords = pickle_object['contours']
        self.imcontour = np.copy(self.im)
        self.coords_count += 1
        self.update_plot()
            
    def clear(self):
        self.coords_count = -1
        self.drawmode = None        
        self.coords = []
        self.update_plot()
        
    def undo_draw(self):
        try:
            oldpoint = self.coords[self.coords_count].pop()
            self.redo.append(oldpoint)
        except IndexError:
            pass
        self.update_plot()

    def redo_draw(self):
        try:
            oldpoint = self.redo.pop()
            self.coords[self.coords_count].append(oldpoint)
        except IndexError as e:
            pass
        self.update_plot()
        
    def exit_drawing(self):
        self.drawmode = None
        
    def black_pixel(self):
        self.coords_count += 1
        self.coords.append([])
        self.update_plot()
        self.drawmode = 'black'

    def next_con(self):
        self.coords_count +=1
        if self.coords_count+1 > len(self.coords):
            self.coords_count -= 1
        self.update_plot()
    def prev_con(self):
        self.coords_count -=1
        if self.coords_count < 0:
            self.coords_count += 1
        self.update_plot()
            
    def next_image(self):
        if self.coords != []:
            MsgBox = tk.messagebox.askokcancel("Proceed?","The contours will be deleted")
            if MsgBox == False:
                return
        self.coords_count = -1
        self.drawmode = None
        self.coords = []
        try:
            self.index+=1
            file = self.flist[self.index]
        except:
            self.index -=1
        self.open_image()
        
    def prev_image(self):
        if self.coords != []:
            MsgBox = tk.messagebox.askokcancel("Proceed?","The contours will be deleted")
            if MsgBox == True:
                return
        self.coords_count = -1
        self.drawmode = None
        self.coords = []
        try:
            self.index-=1
            if self.index <0:
                self.index+=1
                return
            file = self.flist[self.index]
        except:
            self.index +=1
        self.open_image()
        
    def save_contour(self):
        parameters=dict()
        self.filename = self.image_name
        for line in self.params:
            text=str(self.params[line].get())
            parameters[line] = text
            self.filename += text
        
        self.filename +='.pickle'
        parameters['contours'] = self.coords
        parameters['Original_image'] = self.im
        with open('Database/'+self.filename, 'wb') as file:
            pickle.dump(parameters,file)
##        self.coords = []
        self.next_image()
    
    def open_image(self):
        try:
            file = self.flist[self.index]
            self.im = cv.imread(file)
            _,self.image_name = os.path.split(file)
        except IndexError as e:
            print(traceback.print_exc())
            self.im = 255 * np.ones(shape=[1200, 1600, 3], dtype=np.uint8)
        self.imcontour = np.copy(self.im)
        self.update_plot()
        
        
    def update_plot(self):
        self.imcontour = np.copy(self.im)
        for c,i in enumerate(self.coords):
            points = np.array(i)
            colr = (12,55,60)
            if c == self.coords_count:
                colr = (0,0,255)
            try:
                cv2.polylines(self.imcontour,np.int32([points]),True,color=colr,thickness=2)
            except Exception as e:
                print(traceback.format_exc())
        self.imcontour = cv2.cvtColor(self.imcontour, cv2.COLOR_BGR2RGB)
        self.image.set_data(self.imcontour)
        self.canvas.draw()
        
    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)
        self.toolbar.update()

    def key_bindings(self, event):
##        print(event.keysym)
        if event.keysym == 'z':
            self.undo_draw()
        if event.keysym == 'r':
            self.redo_draw()
        if event.keysym == 'Right':
            self.next_image()
        if event.keysym == 'Left':
            self.prev_image()
        if event.keysym == 'p':
            self.drawmode == None
        if event.keysym == 'Down':
            self.prev_con()
        if event.keysym == 'Up':
            self.next_con()
        if event.keysym == 'n':
            self.black_pixel()
            
    def callback(self,event):
        if self.drawmode != None:
            self.redo = []
            if event.xdata == None or event.ydata == None:
                return
            x,y = event.xdata, event.ydata
            
            height,width = self.im.shape[0], self.im.shape[1]
            x = int(round(x*(width/self.ymax)))
            y = int(round(y*(height/self.xmax)))
            
##            print("clicked at", event.xdata, event.ydata,round(event.xdata))
##            print(x,y)
            self.coords[self.coords_count].append((x,y))
            self.update_plot()

    def make_list(self):
        for f in ['Input','Database']:
            if self.dirs==['Input',] and not os.path.isdir(f):
                os.makedirs(f)

        self.flist = []
        for folder in self.dirs:
            try:
                for f in os.listdir(folder):
                    if not os.path.isfile(f):
                        self.dirs.append(folder+'/'+f)
            except:
                if folder.endswith('.tif') or folder.endswith('.jpg'):
                    self.flist.append(folder)
        self.index=0


if __name__ == '__main__':
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', exit)
    app = Make_Contours(root)
    root.mainloop()
