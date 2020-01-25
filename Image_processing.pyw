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
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



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
        self.display_distance = False
        self.displayfocus = False
        self.focus_pts = [(250,135),(1000,900)]
        self.expected_cellsize = 200000
        self.scale = 1
        self.parent.bind('<Key>',self.key_bindings)
        s=ttk.Style()
##        print(s.theme_names())
        s.theme_use('xpnative')
        self.frame09=tk.Frame(master=parent)
        self.frame09.grid(column=0,row=0)
        self.frame0=tk.Frame(master=parent)
        self.frame0.grid(column=0,row=1)
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
        filemenu.add_command(label="Refresh", command=self.refresh_file)
        filemenu.add_command(label="Open", command=self.open_file)
        filemenu.add_command(label="Save", command=self.save_contour)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        parent.config(menu=menubar)

        drawmenu = tk.Menu(menubar, tearoff=0)
        drawmenu.add_command(label="Create Closed Contour", command=self.black_pixel)
        drawmenu.add_command(label="Pause (shortcut='p')", command=self.exit_drawing)
        drawmenu.add_command(label="Resume", command=self.resume_drawing)
        drawmenu.add_command(label="Delete Contour", command=self.delete_contour)
        drawmenu.add_separator()
        drawmenu.add_command(label='Next Image (Right)',command=self.next_image)
        drawmenu.add_command(label='Previous Image (Left)',command=self.prev_image)
        drawmenu.add_command(label="Undo (shortcut=z)", command=self.undo_draw)
        drawmenu.add_command(label="Redo (shortcut=z)", command=self.redo_draw)
        drawmenu.add_command(label='Next Countour (Up)',command=self.next_con)
        drawmenu.add_command(label='Previous Countour (Down)',command=self.prev_con)
        menubar.add_cascade(label="Edit", menu=drawmenu)
        parent.config(menu=menubar)

        settingsmenu = tk.Menu(menubar, tearoff=0)
        settingsmenu.add_separator()
        menubar.add_cascade(label="settings", menu=settingsmenu)
        parent.config(menu=menubar)

        
        conmenu = tk.Menu(menubar, tearoff=0)
        conmenu.add_command(label="Generate Contours", command=self.generate_contours)
        menubar.add_cascade(label="Generate Contours", menu=conmenu)
        parent.config(menu=menubar)

        measmenu = tk.Menu(menubar, tearoff=0)
        measmenu.add_command(label="Distance", command=self.measure_distance)
        measmenu.add_command(label="Calibrate", command=self.calibrate)
        measmenu.add_command(label="Show Focus Area", command=self.show_focus)
        measmenu.add_command(label="Auto-Measure", command=self.auto_measure)
        measmenu.add_command(label="Auto-Measure ALL", command=self.auto_measure_all)
        measmenu.add_separator()
        menubar.add_cascade(label="Measure", menu=measmenu)

        self.image_strvar = tk.StringVar()
        self.image_strvar.set('test')
        tk.Label(master=self.frame0, textvariable = self.image_strvar).grid(column=3,row=3)

        
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
        self.e1,self.e2,self.e3,self.e4 = tk.Entry(self.frame2, width=15),tk.Entry(self.frame2,width=9),tk.Entry(self.frame2,width=9),tk.Entry(self.frame2,width=9)
        self.e1.grid(row=1,column=2)
        self.e2.grid(row=1,column=4)
        self.e3.grid(row=1,column=6)
        self.e4.grid(row=1,column=8)
        tk.Label(self.frame2,text='Name').grid(column=1,row=1)
        tk.Label(self.frame2,text='Material').grid(column=3,row=1)
        tk.Label(self.frame2,text='Curing Time').grid(column=5,row=1)
        tk.Label(self.frame2,text='Intensity (before,after)').grid(column=7,row=1)
        self.params = dict(name=self.e1,material=self.e2,Print_Time=self.e3,Intensity=self.e4)

        
        self.prev = tk.Button(self.frame4,text='Previous',command=self.prev_image)
        self.prev.grid(row=0,column=3)
        self.next = tk.Button(self.frame4,text='Next',command=self.next_image)
        self.next.grid(row=0,column=4)
##        tk.Label(self.frame4,text=' Thresh: ').grid(row=0,column=5)

        self.threshslider = tk.Scale(self.frame4,label='Thresh',from_=0, to=255, orient=tk.HORIZONTAL, resolution=1)
        self.threshslider.set(100)
        self.threshslider.grid(row=0,column=6)
        tk.Label(self.frame4,text=' Contour Area: ').grid(row=0,column=7)
        self.low_area,self.high_area = tk.StringVar(),tk.StringVar()
        self.low_area.set("30000")
        self.area_spinbox1 = tk.Spinbox(self.frame4, from_=0.0, to=1e16,width=10, increment=100,textvariable=self.low_area)
        self.area_spinbox1.grid(row=0, column=8)
        tk.Label(self.frame4,text=' To: ').grid(row=0,column=9)
        self.high_area.set("1500000")
        self.area_spinbox2 = tk.Spinbox(self.frame4, from_=1.0, to=1e16,width=10, increment=1000,textvariable=self.high_area)
        self.area_spinbox2.grid(row=0, column=10)
        
        
        blackbutton = tk.Button(self.frame4,text='New Contour',command=self.black_pixel)
        blackbutton.grid(column=3,row=1)
        undobutton = tk.Button(self.frame4,text='Detect Line',command=self.detect_line)
        undobutton.grid(column=4,row=1)
        drawingexit = tk.Button(self.frame4,text='Pause',command=self.exit_drawing)
        drawingexit.grid(column=5,row=1)
        savedrawing = tk.Button(self.frame4,text='Save',command=self.save_contour)
        savedrawing.grid(column=6,row=1)
        tk.Button(self.frame4,text='Clear',command=self.clear).grid(column=7,row=1)
        tk.Button(self.frame4,text='Generate Contours',command=self.generate_contours).grid(column=8,row=1)
        tk.Button(self.frame4,text='Home',command=self.go_home).grid(column=9,row=1)
        tk.Button(self.frame4,text='Measure',command=self.auto_measure).grid(column=10,row=1)
        tk.Button(self.frame4,text='Hammerhead',command=self.hammerhead).grid(column=11,row=1)
        self.coords = []
        self.redo = []
        self.make_list()
        self.open_image()


        #Testing
        self.generate_contours()
        self.auto_measure()

    def show_focus(self):
        self.displayfocus = True

        self.update_plot()
        
    def hammerhead(self):
        self.generate_contours()
##        print(len(self.coords))
        for i in self.coords:
            miny = max(i, key = lambda t: t[1])
            minx = max(i, key = lambda t: t[0])
            maxy = min(i, key = lambda t: t[1])
            maxx = min(i, key = lambda t: t[0])
            ni = np.array(i)
            rect = cv.minAreaRect(ni)
            ((cx,cy), (width, height), angle_rotation) = rect
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(self.imcontour,[box],0,(0,0,255),2)            
            ##Important fit line function
##            [vx,vy,x,y] = cv2.fitLine(ni,cv2.DIST_L2,0,0.01,0.01)       
##            print([vx,vy,x,y])
##            cv2.line(self.imcontour,(self.imcontour.shape[1]-1,int(((self.imcontour.shape[1]-x)*vy/vx)+y)),(0,int((-x*vy/vx) + y)),255,2)
##            for c,j in enumerate(i):
##                if ((j[0]-i[c-1][0])**2+(j[1]-i[c-1][1])**2)**0.5 >10:
##                    print(c,j)
##                    print(((j[0]-i[c-1][0])**2+(j[1]-i[c-1][1])**2)**0.5)
##        print('minx:',minx,'miny:',miny,'maxx:',maxx,'maxy:',maxy)
        cv2.circle(self.imcontour,minx,radius=6,thickness=6,color=(0,0,255))
        cv2.circle(self.imcontour,miny,radius=6,thickness=6,color=(0,0,255))
        cv2.circle(self.imcontour,maxx,radius=6,thickness=6,color=(0,0,255))
        cv2.circle(self.imcontour,maxy,radius=6,thickness=6,color=(0,0,255))
        self.image.set_data(self.imcontour)
        self.canvas.draw()
##        self.update_plot()
        
    def detect_line(self):
        img = 255 * np.ones(self.imcontour.shape, np.uint8)
        for c,i in enumerate(self.coords):
            points = np.array(i)
            cv2.polylines(img,np.int32([points]),True,color=(0,0,255),thickness=2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,self.threshslider.get(),150,apertureSize = 3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

        self.image.set_data(img)
        self.canvas.draw()
        
    def auto_measure_all(self):
        with open('dimension_data.csv', 'w') as file:
            file.write('Image Name, L1, L2, L3, M1, M2, M3, R1, R2, R3\n')
        for i in self.flist:
##            self.fig.savefig(f'{self.image_name}1.png')
            self.generate_contours()
            self.auto_measure()
            self.next_image()
##            self.canvas.after(50)

    def auto_measure(self):
        yval,horzlines = {},{}
        horzlines, vertlines = {},{}
        horzlines2, vertlines2 = {},{}
        self.scale=5.952380952380952 #Test Microns/pixel for Zeiss V20 microscope
##        print('number of contours: ',len(self.coords))

        contours = []
        other_contours = []
        for i in self.coords:
            ni = np.array(i)
            M = cv.moments(ni)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            area = cv.contourArea(ni)
            if self.focus_pts[0][0]<cx<self.focus_pts[1][0] and self.focus_pts[0][1]<cy<self.focus_pts[1][1]:
                if area > self.expected_cellsize:
                    other_contours.append(i)
                    continue
                contours.append(i)
            else:
                other_contours.append(i)
##        print(len(other_contours))
        average = []
        for i in contours:
            for c,p in enumerate(i):
                average.append(p[0])
        average = np.mean(average)
        
        
        for i in contours:
            M = cv.moments(np.array(i))
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(self.imcontour,(cx,cy),radius=6,thickness=6,color=(255,0,255))
            ni = np.array(i)
            rect = cv.minAreaRect(ni)
            #Angle_rotation starts at -90, and increases up to 0 as it's rotated clockwise
            ((cx,cy), (width, height), angle_rotation) = rect
##            print('avg',average, 'angle: ', angle_rotation)
            newavg = np.mean([f[0] for f in i])
##            print('newavg',newavg)
            if newavg > average:
                side = 'R'
            if newavg<= average:
                side='L'
            data = np.array(i)
            sortIndices = np.argsort(data[:,1])
                        
            miny = max(i, key = lambda t: t[1])
            minx = max(i, key = lambda t: t[0])
            maxy = min(i, key = lambda t: t[1])
            maxx = min(i, key = lambda t: t[0])
            ytargets = {'1':miny[1]-85, '2':(miny[1]+maxy[1])/2, '3':maxy[1]+85}
            xtargets = {'4':(minx[0]+maxx[0])/2}
            
            
##            print(ytargets)
##            print(target)
            group = []
            space = 40          
            
            for c,p in enumerate(i):
                try:
                    p2 = i[c+1]
                except IndexError:
                    p2 = i[-1]
                x,y = p
                x2,y2 = p2
                dist =  ((x-x2)**2+(y-y2)**2)**0.5
                #important loop because find_contour only generates nodes, points along a straight line are not included
                for x,y in zip(np.linspace(x,x2,int(dist)),np.linspace(y,y2,int(dist))):
                    x,y = int(x),int(y)
##                    print(x,y)
                    for name,target in ytargets.items():
                        if target-space< y <target+space:
                            
                            key = side+name
    ##                        print(name, p)
                            
                            if key not in yval:
                                yval[key]=[[],[]]
                                yval[key][0].append(p)

                            if key not in vertlines:
                                vertlines[key]= [[[],[]],[[],[]]]
                            xavg = np.mean([f[0] for f in yval[key][0]])
                                
                                
##                            horzlines[key][0].append(x)
##                            horzlines[key][1].append(y)
                            
                            if xavg-space<x<xavg+space:      
                                yval[key][0].append(p)
                                vertlines[key][0][0].append(x)
                                vertlines[key][0][1].append(y)
                            if xavg-space>x or x>xavg+space:
                                yval[key][1].append(p)
                                vertlines[key][1][0].append(x)
                                vertlines[key][1][1].append(y)
                            
                            cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                    for name,target in xtargets.items():
                        if target-space< x <target+space:
    ##                        print(x,(minx[0]+maxx[0])/2)
                            key = side+name
                            if key not in horzlines:
                                horzlines[key]=[[],[],[],[]]
                                horzlines[key][0].append(p[1])
                            
                            cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                            for c, group in enumerate(horzlines[key]):
                                if horzlines[key][c] == []:
                                    horzlines[key][c].append(p[1])
                                    break
                                yavg = np.mean(horzlines[key][c])
                                if yavg-space<y<yavg+space:
                                    try:
                                        horzlines[key][c].append(p[1])
                                    except AttributeError as e:
                                        print(traceback.print_exc())
                                    break
                                
            for other_contour in other_contours:
                for c,po in enumerate(other_contour):
                    try:
                        po2 = other_contour[c+1]
                    except IndexError:
                        po2 = other_contour[-1]
                    x,y = po
                    x2,y2 = po2
                    dist =  ((x-x2)**2+(y-y2)**2)**0.5
                    for x,y in zip(np.linspace(x,x2,int(dist)),np.linspace(y,y2,int(dist))):
                        x,y = int(x),int(y)
                        for name,target in ytargets.items():
                            if target-space< y <target+space:
                                if side=='L':
                                    sideval=min([np.mean([f[0][0] for f in vertlines['L'+name]]),np.mean([f[1][0] for f in vertlines['L'+name]])])
                                    if x<=sideval:
                                        if x<=10:
                                            #Checks that the contour is not the edge of the image
                                            continue
                                        key = 'LG'+name
                                        if key not in vertlines2:
                                            vertlines2[key] = [[],[]]
                                            vertlines2[key][0].append(x)
                                            vertlines2[key][1].append(y)
                                        avg2 = np.mean([f for f in vertlines2[key][0]])
                                        if x<avg2-space:
                                            vertlines2[key] = [[],[]]
                                            vertlines2[key][0].append(x)
                                            vertlines2[key][1].append(y)
                                        avg2 = np.mean([f for f in vertlines2[key][0]])
                                        if x>avg2+space:
                                            continue
                                        vertlines2[key][0].append(x)
                                        vertlines2[key][1].append(y)
##                                        cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                                if side=='R':
                                    sideval=max([np.mean([f[0][1] for f in vertlines['R'+name]]),np.mean([f[1][1] for f in vertlines['R'+name]])])
                                    if x>=sideval:
                                        if x>=self.imcontour.shape[1]-10:
                                            continue
                                        key = 'RG'+name
                                        if key not in vertlines2:
                                            vertlines2[key] = [[],[]]
                                            vertlines2[key][0].append(x)
                                            vertlines2[key][1].append(y)
                                        avg2 = np.mean([f for f in vertlines2[key][0]])
                                        if x<avg2-space:
                                            vertlines2[key] = [[],[]]
                                            vertlines2[key][0].append(x)
                                            vertlines2[key][1].append(y)
                                        avg2 = np.mean([f for f in vertlines2[key][0]])
                                        if x>avg2+space:
                                            continue
                                        vertlines2[key][0].append(x)
                                        vertlines2[key][1].append(y)

                        for name,target in xtargets.items():
                            if target-space< x <target+space:
                                if side=='L':
                                    if y<=maxy[1]:
                                        if y<=10:
                                            continue
                                        key = 'LT'
                                        if key not in horzlines2:
                                            horzlines2[key] = [[],[]]
                                            horzlines2[key][0].append(x)
                                            horzlines2[key][1].append(y)
                                        cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                                    if y>=miny[1]:
                                        if y>=self.imcontour.shape[1]-10:
                                            continue
                                        key = 'LB'
                                        if key not in horzlines2:
                                            horzlines2[key] = [[],[]]
                                            horzlines2[key][0].append(x)
                                            horzlines2[key][1].append(y)
                                        cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                                if side=='R':
                                    if y<=maxy[1]:
                                        if y<=10:
                                            continue
                                        key = 'RT'
                                        if key not in horzlines2:
                                            horzlines2[key] = [[],[]]
                                            horzlines2[key][0].append(x)
                                            horzlines2[key][1].append(y)
                                        cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                                    if y>=miny[1]:
                                        if y>=self.imcontour.shape[1]-10:
                                            continue
                                        key = 'RB'
                                        if key not in horzlines2:
                                            horzlines2[key] = [[],[]]
                                            horzlines2[key][0].append(x)
                                            horzlines2[key][1].append(y)
                                        cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
            for key,p in vertlines2.items():
                for x,y in zip(p[0],p[1]):
                    cv2.circle(self.imcontour,(x,y),radius=3,thickness=2,color=(0,150,255))
                    
            yval2 = {}
            for i in yval:
                if i not in yval2:
                    yval2[i] = ['','']
                yval2[i][0] = [np.mean([f[0] for f in yval[i][0]]),np.mean([f[1] for f in yval[i][0]])]
                
                yval2[i][1] = [np.mean([f[0] for f in yval[i][1]]),np.mean([f[1] for f in yval[i][1]])]
##                try:
##                    print(int(yval2[i][1][0]))
##                except:
##                    print(yval)
            for key,val in horzlines.items():
##                print(val)
                for c,group in enumerate(val):
                    val[c] = np.mean(group)

                       
                
##            print('minx:',minx,'miny:',miny,'maxx:',maxx,'maxy:',maxy)
            new = (miny[0],miny[1]-85)

            
                
            
##            cv2.circle(self.imcontour,minx,radius=6,thickness=6,color=(0,0,255))
##            cv2.circle(self.imcontour,miny,radius=6,thickness=6,color=(0,0,255))
##            cv2.circle(self.imcontour,maxx,radius=6,thickness=6,color=(0,0,255))
##            cv2.circle(self.imcontour,maxy,radius=6,thickness=6,color=(0,0,255))
##            cv2.circle(self.imcontour,p2,radius=6,thickness=6,color=(0,0,255))
##        for key,([x3,y3],[x1,y1]) in vertlines.items():
##            for x,y in ([x3,y3],[x1,y1]):
##                x = np.array(x).reshape(-1, 1)
##                y = np.array(y).reshape(-1, 1)
##                x1= np.array(x1).reshape(-1, 1)
##                y1 = np.array(y1).reshape(-1, 1)
##                regr = linear_model.LinearRegression()
##                regr.fit(x,y)
##                y_pred = regr.predict(x)
##    ##            print(y_pred)
##    ##            print(y)
##                coef,inter = regr.coef_, regr.intercept_
##                print(key)
##                print('Coefficients: ', coef,inter)
##                print('Mean squared error: %.2f'% mean_squared_error(y, y_pred))
##                print('Coefficient of determination: %.2f'% r2_score(y, y_pred))
##                pf1 = (0,inter)
##                pf2 = (10000,inter+10000*coef)
##                cv2.line(self.imcontour, pf1,pf2, (0,0,255), 2)
##                print((x[0],y_pred[0]),(x[-1],y_pred[-1]))
##                cv2.line(self.imcontour, (int(min(x)),int(min(y_pred))),(int(max(x)),int(max(y_pred))), (0,0,255), 5)
##            for w,z,z_pred in zip(x,y,y_pred):
##                cv2.circle(self.imcontour,(w,z_pred),radius=3,thickness=2,color=(0,0,255))
##                self.image.set_data(self.imcontour)
##                self.canvas.draw()
        
        meas_lines = {}
        self.image.set_data(self.imcontour)
        self.canvas.draw()
        try:
            L1 = abs(yval2['L1'][0][0]-yval2['L1'][1][0])*self.scale
            meas_lines['L1'] = [(int(yval2['L1'][0][0]),int(yval2['L1'][0][1])),(int(yval2['L1'][1][0]),int(yval2['L1'][0][1]))]
            L2 = abs(yval2['L2'][0][0]-yval2['L2'][1][0])*self.scale
            meas_lines['L2'] = [(int(yval2['L2'][0][0]),int(yval2['L2'][0][1])),(int(yval2['L2'][1][0]),int(yval2['L2'][0][1]))]
            L3 = abs(yval2['L3'][0][0]-yval2['L3'][1][0])*self.scale
            meas_lines['L3'] = [(int(yval2['L3'][0][0]),int(yval2['L3'][0][1])),(int(yval2['L3'][1][0]),int(yval2['L3'][0][1]))]
            R1 = abs(yval2['R1'][0][0]-yval2['R1'][1][0])*self.scale
            meas_lines['R1'] = [(int(yval2['L1'][0][0]),int(yval2['L1'][0][1])),(int(yval2['R1'][1][0]),int(yval2['L1'][0][1]))]
            R2 = abs(yval2['R2'][0][0]-yval2['R2'][1][0])*self.scale
            meas_lines['R2'] = [(int(yval2['L2'][0][0]),int(yval2['L2'][0][1])),(int(yval2['R2'][1][0]),int(yval2['L2'][0][1]))]
            R3 = abs(yval2['R3'][0][0]-yval2['R3'][1][0])*self.scale
            meas_lines['R3'] = [(int(yval2['L3'][0][0]),int(yval2['L3'][0][1])),(int(yval2['R3'][1][0]),int(yval2['L3'][0][1]))]
            M1 = abs(yval2['L1'][0][0]-yval2['R1'][1][0])*self.scale
            meas_lines['M1'] = [(int(yval2['L1'][0][0]),int(yval2['L1'][0][1])),(int(yval2['R1'][1][0]),int(yval2['L1'][0][1]))]
            M2 = abs(yval2['L2'][0][0]-yval2['R2'][1][0])*self.scale
            meas_lines['M2'] = [(int(yval2['L2'][1][0]),int(yval2['L2'][0][1])),(int(yval2['R2'][0][0]),int(yval2['L2'][0][1]))]
            M3 = abs(yval2['L3'][0][0]-yval2['R3'][1][0])*self.scale
            meas_lines['M3'] = [(int(yval2['L3'][0][0]),int(yval2['L3'][0][1])),(int(yval2['R3'][1][0]),int(yval2['L3'][0][1]))]
        except ValueError as e:
            print(e,'Value Error')
        
##        print(f'{L1}, {L2}, {L3}, {M1}, {M2}, {M3}, {R1}, {R2}, {R3}')
        with open('dimension_data.csv', 'a') as file:
            
            file.write(f'{self.image_name}, {L1}, {L2}, {L3}, {M1}, {M2}, {M3}, {R1}, {R2}, {R3}')
            file.write('\n')
        for key,line in meas_lines.items():
            cv2.line(self.imcontour,line[0],line[1],color=(0,0,255),thickness=2)
        self.image.set_data(self.imcontour)
        self.canvas.draw()
##        self.fig.savefig(f'{self.image_name}2.png')
        
    def go_home(self):
        self.ax.relim()
        self.ax.autoscale()
        self.canvas.draw()
        
    def calibrate(self):
        self.drawmode = 'calibrate'
        self.calpoint = []
    def open_file(self):
        from tkinter import filedialog
        filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("pickle files","*.pickle"),("all files","*.*")))
        print(filename)
        with open(filename,"rb") as file:
            pickle_object = pickle.load(file)
        self.im = pickle_object['Original_image']
        self.coords = pickle_object['contours']
        self.coords_count += 1
        
        self.update_plot()
            
    def clear(self):
        self.coords_count = -1
        self.drawmode = None        
        self.coords = []
        self.update_plot()

    def generate_contours(self):
        im_bw = cv2.cvtColor(self.im, cv2.COLOR_RGB2GRAY) #Needs to be converted to black and white
        ret, thresh = cv.threshold(im_bw, self.threshslider.get(),255,cv2.THRESH_BINARY)
        contours, heirarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = [i for i in contours if int(self.high_area.get()) > cv.contourArea(i) and cv.contourArea(i)>int(self.low_area.get())]
##        cv2.imshow('image', self.image)
##        cv2.waitKey(0)
        for contour in contours:
##            area = cv2.contourArea(contour)
##            print(area)
            format_contour = []
            for i in contour:
                (x,y) = i[0][0],i[0][1]
                format_contour.append((x,y))
            self.coords.append(format_contour)
        self.update_plot()
        
    def delete_contour(self):
##        if self.coords != []:
##            MsgBox = tk.messagebox.askokcancel("Proceed?","This contour will be deleted")
##            if MsgBox == False:
##                return
        del self.coords[self.coords_count]

        if self.coords_count > 0:
            self.coords_count -= 1
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
    def resume_drawing(self):
        self.drawmode = 'black'
        
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
##        if self.coords != []:
##            MsgBox = tk.messagebox.askokcancel("Proceed?","The contours will be deleted")
##            if MsgBox == False:
##                return
        self.coords_count = -1
        self.drawmode = None
        self.coords = []
        try:
            self.index+=1
            if self.index >= len(self.flist):
                self.index-=1
                return
##            file = self.flist[self.index]
        except:
            self.index -=1
        self.open_image()
        
    def prev_image(self):
##        if self.coords != []:
##            MsgBox = tk.messagebox.askokcancel("Proceed?","The contours will be deleted")
##            if MsgBox == False:
##                return
        self.coords_count = -1
        self.drawmode = None
        self.coords = []
        try:
            self.index-=1
            if self.index <0:
                self.index+=1
                return
##            file = self.flist[self.index]
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
##        tk.messagebox.showinfo("Successful Save","Contours have been saved.")

    
    def open_image(self):
        try:
            file = self.flist[self.index]
            self.im = cv.imread(file)
            self.image_name = os.path.split(file)[1].split('.')[0]   #filename
            self.image_strvar.set(file)
            
        except IndexError as e:
            print(traceback.print_exc())
            self.im = 255 * np.ones(shape=[1200, 1600, 3], dtype=np.uint8)
        self.imcontour = np.copy(self.im)
        self.update_plot()
        
        
    def update_plot(self):
        self.imcontour = np.copy(self.im)
        for c,i in enumerate(self.coords):
            points = np.array(i)
            colr = (0,255,0)
            if c == self.coords_count:
                colr = (0,0,255)
            try:
                cv2.polylines(self.imcontour,np.int32([points]),True,color=colr,thickness=2)
            except Exception as e:
                print(traceback.format_exc())
##        if self.display_distance == True:
##            
##            d = ((self.mp[1][0]-self.mp[0][0])**2+(self.mp[0][1]-self.mp[1][1])**2)**1/2
##            d = round(d*self.scale,4)
##            mx,my = ((self.mp[1][0]+self.mp[0][0])/2,(self.mp[0][1]+self.mp[1][1])/2)
##            cv2.putText(self.imcontour,f'{d}',(int(mx-80), int(my-10)),cv2.FONT_HERSHEY_SIMPLEX, 1.85, (0,0,0), 2)
##            print(mx,my)
##            print('distance: ', d)
##            cv2.line(self.imcontour, (self.mp[0][0], self.mp[0][1]), (self.mp[1][0], self.mp[1][1]), (0,0,255), 5)
##            self.display_distance = False
        if self.displayfocus == True:
            cv2.rectangle(self.imcontour, self.focus_pts[0], self.focus_pts[1],(86,35,211),2)
        self.imcontour = cv2.cvtColor(self.imcontour, cv2.COLOR_BGR2RGB)
        self.image.set_data(self.imcontour)
        self.canvas.draw()
        
    def on_key_press(self, event):
        key_press_handler(event, self.canvas, self.toolbar)
        self.toolbar.update()
##        self.toolbar.push_current()

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
            self.drawmode = None
        if event.keysym == 'Down':
            self.prev_con()
        if event.keysym == 'Up':
            self.next_con()
        if event.keysym == 'n':
            self.black_pixel()

    def measure_distance(self):
        self.drawmode = 'linear_measure'
        self.mp = []
        
    def callback(self,event):
        if self.drawmode == 'black':
            self.redo = []
            if event.xdata == None or event.ydata == None:
                return
            x,y = event.xdata, event.ydata
            
            height,width = self.im.shape[0], self.im.shape[1]
            x = int(round(x*(width/self.ymax)))
            y = int(round(y*(height/self.xmax)))
            
            print("clicked at",round(x),round(y))
##            print(x,y)
            try:
                self.coords[self.coords_count].append((x,y))
            except AttributeError:
                np.append(self.coords[self.coords_count],(x,y))
            self.update_plot()
            
        if self.drawmode == 'linear_measure':
            if event.xdata == None or event.ydata == None:
                return
            x,y = event.xdata, event.ydata
            height,width = self.im.shape[0], self.im.shape[1]
            x = int(round(x*(width/self.ymax)))
            y = int(round(y*(height/self.xmax)))
            print(x,y)
            if len(self.mp) <2:
                self.mp.append((x,y))
            if len(self.mp) >= 2:
                
##                self.display_distance = True
##                self.update_plot()
                d = ((self.mp[1][0]-self.mp[0][0])**2+(self.mp[0][1]-self.mp[1][1])**2)**0.5
                d = round(d*self.scale,4)

                mx,my=((self.mp[1][0]+self.mp[0][0])/2,(self.mp[0][1]+self.mp[1][1])/2)
                cv2.putText(self.imcontour,f'{d}',(int(mx-80), int(my-10)),cv2.FONT_HERSHEY_SIMPLEX, 1.85, (0,255,255), 2)
                cv2.line(self.imcontour, (self.mp[0][0], self.mp[0][1]), (self.mp[1][0], self.mp[1][1]), (0,0,255), 5)
                self.image.set_data(self.imcontour)
                self.canvas.draw()
                self.mp = []
                
        if self.drawmode == 'calibrate':
            if event.xdata == None or event.ydata == None:
                return
            x,y = event.xdata, event.ydata
            height,width = self.im.shape[0], self.im.shape[1]
            x = int(round(x*(width/self.ymax)))
            y = int(round(y*(height/self.xmax)))
            if len(self.calpoint) <2:
                self.calpoint.append((x,y))
            if len(self.calpoint) >= 2:
                from tkinter.simpledialog import askstring
                d = ((self.calpoint[1][0]-self.calpoint[0][0])**2+(self.calpoint[0][1]-self.calpoint[1][1])**2)**0.5
                try:
                    true_distance = float(askstring('Scale', 'What is the calibration scale?'))
                    self.drawmode = None
                except TypeError as e:
##                    tk.messagebox.showwarning("Warning","Invalid input")
                    self.calpoint=[]
                    return
                self.scale = true_distance/d
                print('scale:', self.scale)
    def refresh_file(self):
        self.make_list()
        self.open_image()
        self.update_plot()

        
    def make_list(self):
        for f in ['Input','Database']:
            if self.dirs==['Input',] and not os.path.isdir(f):
                os.makedirs(f)

        self.flist = []
        self.dirs=['Input']
        for folder in self.dirs:
            try:
                for f in os.listdir(folder):
                    if not os.path.isfile(f):
                        self.dirs.append(folder+'/'+f)
            except:
                if folder.endswith('.tif') or folder.endswith('.jpg')or folder.endswith('.png'):
                    self.flist.append(folder)
        self.index=0


if __name__ == '__main__':
    root = tk.Tk()
    root.iconbitmap('resources/uml_logo.ico')
    root.protocol('WM_DELETE_WINDOW', exit)
    app = Make_Contours(root)
    root.mainloop()
