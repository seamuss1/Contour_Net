import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
import os

class Graph_Results:
    def __init__(self, file=None):
        self.file = file
        self.data = pd.read_csv(self.file,index_col=0)
        self.plotdic = {}
        self.folder = 'sample104_2-19-20_results'
        if self.folder not in [f for f in os.listdir()]:
            os.makedirs(self.folder)
        
        self.dimension_key = {'L1':1700,
                             'L2':500,
                             'L3':1700,
                             'R1':1700,
                             'R2':500,
                             'R3':1700,
                             'M1':400,
                             'M2':2800,
                             'M3':400,
                             'LG1':1200,
                             'LG2':1200,
                             'LG3':1200,
                             'RG1':1200,
                             'RG2':1200,
                             'RG3':1200,
                             'RT':1200,
                             'LT':1200,
                             'RB':1200,
                             'LB':1200,
                             'LV1':1050,
                             'LV2':1700,
                             'LV3':1050,
                             'RV1':1050,
                             'RV2':1700,
                             'RV3':1050
                             }
                             
    def get_plotdic(self):
        dic = {}
        for key in self.data:
            for c,i in enumerate(self.data[key]):
                file = self.data.index[c]
                if file=='Average:':
                    continue
                if file not in dic:
                    dic[file] = {}
                if key not in dic[file]:
                    dic[file][key] = []
                try:
                    diff = float(i) - self.dimension_key[key]
                except:
                    diff = 0
                dic[file][key].append(diff)
        dic2 = {}
        dic3 = {}
        for key,value in dic.items():
            avg = []
            name = key.split('/')[-1].split('.')[0]
            for key2,value2 in value.items():
                try:
                    avg2 = np.mean(value2)
                    if key2 not in dic3:
                        dic3[key2]={}
                    dic3[key2][name] = avg2
                except:
                    continue
                avg.append(avg2)
            avg = np.mean(avg)
            dic2[name] = avg

##        plt.ion()
        for name,dictionary in dic3.items():
##            self.fig = plt.figure()
##            ax = Axes3D(self.fig)
            x = []
            y = []
            dz=[]
            z = []
            dx = []
            dy = []
            c1,c2=1,1
            for key,value in dictionary.items():
                x.append(c1)
                y.append(c2)
                dz.append(value)
                z.append(0)
                dx.append(0.5)
                dy.append(0.5)
                c1+=1
                if c1==6:
                    c1=1
                    c2+=1
            x,y,dz = np.array(x),np.array(y),np.array(dz)
##            labels = [item.get_text() for item in ax.get_xticklabels()]
##            labels = ['A','B','C','D','E']
##            plt.yticks(np.arange(min(y), max(y)+1, 1.0))
##            plt.xticks(np.arange(min(x), max(x)+1, 1.0))
##            
##            ax.set_title('{} Design Value = {} um'.format(name,self.dimension_key[name]))
##    ##        ax.set_xticklabels(labels)
##            ax.set_yticklabels(labels)
##            ax.set_zlabel('Difference from Spec (um)')
            self.plotdic[name] = (x,y,dz)
##            surf = ax.plot_trisurf(x,y,dz,cmap=cm.coolwarm)
##            self.plotdic[name] = ax.plot_trisurf(x,y,dz,cmap=cm.coolwarm)
    ##        self.ax.autoscale_view()
##            self.fig.colorbar(self.plotdic[name], shrink=0.5, aspect=5)
        return self.plotdic
        
    def plot_position(self):
        
        dic = {}
        for key in self.data:
            for c,i in enumerate(self.data[key]):
                file = self.data.index[c]
                if file=='Average:':
                    continue
                if file not in dic:
                    dic[file] = {}
                if key not in dic[file]:
                    dic[file][key] = []
                try:
                    diff = float(i) - self.dimension_key[key]
                except:
                    diff = 0
                dic[file][key].append(diff)
        dic2 = {}
        dic3 = {}
        for key,value in dic.items():
            avg = []
            name = key.split('/')[-1].split('.')[0]
            for key2,value2 in value.items():
                try:
                    avg2 = np.mean(value2)
                    if key2 not in dic3:
                        dic3[key2]={}
                    dic3[key2][name] = avg2
                except:
                    continue
                avg.append(avg2)
            avg = np.mean(avg)
            dic2[name] = avg

        plt.ion()
        for name,dictionary in dic3.items():
            self.fig = plt.figure()
            ax = Axes3D(self.fig)
            x = []
            y = []
            dz=[]
            z = []
            dx = []
            dy = []
            c1,c2=1,1
            for key,value in dictionary.items():
                x.append(c1)
                y.append(c2)
                dz.append(value)
                z.append(0)
                dx.append(0.5)
                dy.append(0.5)
                c1+=1
                if c1==6:
                    c1=1
                    c2+=1
            x,y,dz = np.array(x),np.array(y),np.array(dz)
            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels = ['A','B','C','D','E']
            plt.yticks(np.arange(min(y), max(y)+1, 1.0))
            plt.xticks(np.arange(min(x), max(x)+1, 1.0))
            
            ax.set_title('{} Design Value = {} um'.format(name,self.dimension_key[name]))
    ##        ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_zlabel('Difference from Spec (um)')
            surf = ax.plot_trisurf(x,y,dz,cmap=cm.coolwarm)
    ##        self.ax.autoscale_view()
            self.fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.draw()
            plt.pause(0.001)
            self.fig.savefig(f'{os.path.join(self.folder,name)}_surfaceplot.png')
            plt.close()


if __name__=='__main__':
    app = Graph_Results('Database/sample104_2-19-20.csv')
    app.plot_position()
