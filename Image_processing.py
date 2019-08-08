import os, time, cv2,datetime
import cv2 as cv
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
import pickle

class Make_Contours:
    def __init__(self, parent, dirs=['Input',],extension='.tif'):
        self.master = parent
        self.dirs = dirs
        self.extension = extension
        self.frame1=tk.Frame(master=parent)
        self.frame1.grid(column=0,row=0)
        self.frame2=tk.Frame(master=parent)
        self.frame2.grid(column=0,row=1)
        self.frame3=tk.Frame(master=parent)
        self.frame3.grid(column=0,row=2)
        
        self.fig,self.ax = plt.subplots(1,1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
        
        self.canvas.get_tk_widget().pack()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame1)
        self.toolbar.update()
        
        self.canvas._tkcanvas.pack()
        self.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.submit = tk.Button(master=self.frame3,text='Save Contour',command=self.submit_contour)
        self.submit.pack()

        tk.Label(self.frame2,text='Threshold Values: ').grid(column=1,row=0)
        self.thresh1,self.thresh2,self.thresh3 = tk.StringVar(),tk.StringVar(),tk.StringVar()
        self.e1,self.e2,self.e3 = tk.Entry(self.frame2,textvariable=self.thresh1,width=5),tk.Entry(self.frame2,textvariable=self.thresh2,width=5),tk.Entry(self.frame2,textvariable=self.thresh3,width=5)
        self.e1.grid(row=0,column=2)
        self.e2.grid(row=0,column=3)
        self.e3.grid(row=0,column=4)
        self.thresh1.set('175')
        self.thresh2.set('255')
        self.thresh3.set('0')

        
        self.refresh = tk.Button(self.frame2,text='Refresh Image',command=self.refresh_image)
        self.refresh.grid(row=0,column=5)
        self.prev = tk.Button(self.frame2,text='Previous',command=self.next_image)
        self.prev.grid(row=0,column=6)
        self.next = tk.Button(self.frame2,text='Next',command=self.next_image)
        self.next.grid(row=0,column=7)
        self.make_list()
        self.update_plot()
    def next_image(self):
        self.index+=1
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
        self.index +=1
        self.update_plot()
    def update_plot(self):

        file = self.flist[self.index]
        self.im = cv.imread(file)
        self.im =  cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)
        t1,t2,t3 = int(self.thresh1.get()),int(self.thresh2.get()),int(self.thresh3.get())        
        
        ret, thresh = cv.threshold(self.im, t1, t2, t3)
        
        self.contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.contours = [i for i in self.contours if 500000 > cv.contourArea(i) and cv.contourArea(i)>2000]
        areas = [cv.contourArea(i) for i in self.contours]
        perimeter = [round(cv.arcLength(i,True),2) for i in self.contours]
        self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
        cv.drawContours(self.im, self.contours, -1, (255,0,0), 2)
        
        self.ax.imshow(self.im)
        self.canvas.draw()
    def update_contours(self, contours):
        print('update contour')
    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)
        self.toolbar.update()
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
                if folder.endswith('.tif'):
                    self.flist.append(folder)
        self.index=0
if __name__ == '__main__':
    root = tk.Tk()
    app = Make_Contours(root)
    root.mainloop()


