import cv2
import numpy as np

def getTiles():

    tile={}

    x_coord=int(418*0.15)
    y_coord=int(np.around(703*0.56))
    
    xcounter=1

    for i in range(1,271):
        tile[i]=(int(x_coord),int(y_coord))
        xcounter+=1
        if xcounter>18:
            x_coord=int(418*0.15)
            y_coord+=13.5
            xcounter=1
        else:
            x_coord+=17.5
    
    return tile

def matchClock(frame,clock):
    
    frame=frame[360:-90,50:-50,:]
    
    clock_gray = cv2.cvtColor(clock, cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    res = cv2.matchTemplate(frame_gray,clock_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6

    loc = np.where( res >= threshold)
    
    _,found=np.shape(loc)
    
    if found>0:
        return loc[0][-1],loc[1][-1] ## y, x
    else:
        return -1,-1

def getTileNumber(x,y,tile):
    
    y+=4
    x+=6
    #Adjust found clock coordinates to represent center of clock
    
    for t in tile:
        tx,ty=tile[t]
        if ((x<tx+9) and (x>tx-9)) and ((y<ty+7) and (y>ty-7)) :
            return t
        
def findClock(frame,clock):

    y,x=matchClock(frame,clock)
    
    tiles=getTiles()
    
    oneHotArray=np.zeros(len(tiles))
    
    if x>0 and y>0:
        y+=360
        x+=50
        oneHotArray[getTileNumber(x,y,tiles)]=1
        return True,oneHotArray
    else:
        return False,oneHotArray
        
