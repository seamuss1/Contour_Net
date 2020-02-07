import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

class Graph_Results:
    def __init__(self, file=None):
        self.file = file
        self.data = pd.read_csv(self.file,index_col=0)
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
                             

    def plot_position(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
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
                diff = float(i) - self.dimension_key[key]
                dic[file][key].append(diff)
        dic2 = {} 
        for key,value in dic.items():
            avg = []
            name = key.split('/')[-1].split('.')[0]
            for key2,value2 in value.items():
                try:
                    avg2 = np.mean(value2)
                except:
                    continue
                avg.append(avg2)
            avg = np.mean(avg)
            dic2[name] = avg

        x = []
        y = []
        dz=[]
        z = []
        dx = []
        dy = []
        c1,c2=0,0
        for key,value in dic2.items():
            x.append(c1)
            y.append(c2)
            dz.append(value)
            z.append(0)
            dx.append(0.5)
            dy.append(0.5)
            c1+=1
            if c1==4:
                c1=0
                c2+=1
        
        self.ax.bar3d(x, y, z, dx, dy, dz, zsort='min',shade=True)
        self.ax.autoscale_view()
        plt.show()


if __name__=='__main__':
    app = Graph_Results('Database/s1ample_68_1-31-20.csv')
    app.plot_position()
