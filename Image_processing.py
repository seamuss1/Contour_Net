import os, time, cv2,datetime
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
        self.master = parent
        parent.title('Contour Extraction')
        self.dirs = dirs
        self.extension = extension
        self.drawmode = None

##        s=ttk.Style()
##        print(s.theme_names())
##        s.theme_use('winnative')
        self.frame1=tk.Frame(master=parent)
        self.frame1.grid(column=0,row=0)
        self.frame2=tk.Frame(master=parent)
        self.frame2.grid(column=0,row=1)
        self.frame3=tk.Frame(master=parent)
        self.frame3.grid(column=0,row=2)
        self.frame4=tk.Frame(master=parent)
        self.frame4.grid(column=0,row=3)

        
        self.fig,self.ax = plt.subplots(1,1)
        #plt.gca().invert_yaxis()
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0.03, bottom=0.07, right=0.98, top=0.97, wspace=0, hspace=0)
        #self.fig.set_size_inches((8,8))
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame1)
        self.toolbar.update()
        self.canvas._tkcanvas.config(bg='blue')
        self.canvas._tkcanvas.pack()
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.callbacks.connect('button_press_event', self.callback)
        self.submit = tk.Button(master=self.frame3,text='Save Contour',command=self.submit_contour)
        self.submit.pack()


        tk.Label(self.frame2,text='Input Parameters: ').grid(column=1,row=1)
        self.e1,self.e2,self.e3 = tk.Entry(self.frame2, width=9),tk.Entry(self.frame2,width=9),tk.Entry(self.frame2,width=9)
        self.e1.grid(row=1,column=2)
        self.e2.grid(row=1,column=3)
        self.e3.grid(row=1,column=4)

        
        self.refresh = tk.Button(self.frame2,text='Refresh Image',command=self.refresh_image)
        self.refresh.grid(row=0,column=5)
        self.prev = tk.Button(self.frame2,text='Previous',command=self.prev_image)
        self.prev.grid(row=0,column=6)
        self.next = tk.Button(self.frame2,text='Next',command=self.next_image)
        self.next.grid(row=0,column=7)

        blackbutton = tk.Button(self.frame4,text='black pixels',command=self.black_pixel)
        blackbutton.grid(column=3,row=1)
        undobutton = tk.Button(self.frame4,text='Undo',command=self.undo_draw)
        undobutton.grid(column=4,row=1)
        drawingexit = tk.Button(self.frame4,text='Exit drawing',command=self.exit_drawing)
        drawingexit.grid(column=5,row=1)
        self.coords = []
        self.make_list()
        self.open_image()
        self.update_plot()
    def undo_draw(self):
        try:
            del self.coords[-1]
        except IndexError:
            pass
        self.update_plot()
    def exit_drawing(self):
        self.drawmode = None
        
    def black_pixel(self):
        self.drawmode = 'black'
        
    def next_image(self):
        try:
            self.index+=1
            file = self.flist[self.index]
        except:
            self.index -=1
            print('Index Error')
        self.open_image()
        self.update_plot()
        
    def prev_image(self):
        try:
            self.index-=1
            file = self.flist[self.index]
        except:
            self.index +=1
            print('Index Error')
        self.open_image()
        self.update_plot()
    
    def refresh_image(self):
        try:
            int(self.thresh1.get()),int(self.thresh2.get()),int(self.thresh3.get())
        except:
            print('Threshold values need to be integers')
            return
        self.update_plot()
    def submit_contour(self):
        try:
            int(self.thresh1.get()),int(self.thresh2.get()),int(self.thresh3.get())
        except:
            print('Threshold values need to be integers')
            return
        self.filename = 'pickle'+str(datetime.datetime.now())[-6:]+'.pickle'
        dic = dict(original_image=self.im,contours=self.contours)
        with open('Database/'+self.filename, 'wb') as file:
            pickle.dump(dic,file)
        try:
            self.index+=1
        except:
            print('Index Error')
        self.open_image()
        self.update_plot()
    
    def open_image(self):
        file = self.flist[self.index]
        self.im = cv.imread(file)
    def update_plot(self):
        self.imcontour = np.copy(self.im)
        points = np.array(self.coords)
        try:
            cv2.polylines(self.imcontour,np.int32([points]),True,color=(0,0,255),thickness=2)
        except Exception as e:
            print(e)
        self.imcontour = cv2.cvtColor(self.imcontour, cv2.COLOR_BGR2RGB)
        
        self.ax.imshow(self.imcontour)
        self.canvas.draw()
    def update_contours(self, contours):
        print('update contour')
    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)
        print(event,event.x,event.y)
        self.toolbar.update()
    def callback(self,event):
        if self.drawmode != None:
            if self.drawmode == 'black':
                self.drawcolor = 0
            if event.x == None or event.y == None:
                return
            self.pensize = 0.5
            x,y = int(round(event.xdata)), int(round(event.ydata))
            self.coords.append((x,y))
##            self.im[y,x] = self.drawcolor
##            for m in range(x-self.pensize,x+self.pensize):
##                for n in range(y-self.pensize,y+self.pensize):
##                    self.im[n,m] = self.drawcolor
            print("clicked at", event.xdata, event.ydata,round(event.xdata))
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



class Page(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self)
    def show(self):
        self.lift()

class Page1(tk.Frame):
   def __init__(self, *args, **kwargs):
       tk.Frame.__init__(self)
       label = tk.Label(self, text="This is page 1")
       label.pack(side="top", fill="both", expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    app = Make_Contours(root)
    root.mainloop()


