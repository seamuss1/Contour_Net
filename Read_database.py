import pickle, os
import matplotlib.pyplot as plt
import numpy as np
import cv2

flist = [f for f in os.listdir('Database')]
for file in flist:
    file = 'Database/'+file
    with open(file,"rb") as file:
        pickle_object = pickle.load(file)
    for i in pickle_object:
        print(pickle_object[i])
    im = pickle_object['Original_image']
    for c,i in enumerate(pickle_object['contours']):
        points = np.int32(np.array(i))
        print(points.shape)
        colr = (12,55,60)

        

        epsilon = 0.001*cv2.arcLength(np.int32([points]),True)
        approx = cv2.approxPolyDP(np.int32([points]),epsilon,True)
        print(approx.shape)
        cv2.polylines(im,[points],True,color=colr,thickness=2)
        cv2.drawContours(im, [approx], -1, (255,0,0), 2)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    plt.show()
