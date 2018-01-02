import numpy as np
import os
import cv2
from PIL import Image

def findCard(frame,template):
    template=template[10:-10,2:-2,:]
    template_gray = cv2.cvtColor(template.astype(np.float32), cv2.COLOR_RGB2GRAY)
    frame_gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(frame_gray,template_gray,cv2.TM_CCOEFF_NORMED)
    threshold = 0.7

    loc = np.where( res >= threshold)
    
    _,found=np.shape(loc)
    
    if found>0:
        return 1,loc[0][-1],loc[1][-1] 
    else:
        return 0,-1,-1

def cardDetector(frame,card_templates):
            
    deck_crop = frame[590:675,50:280]
    
    oneHotArray=np.zeros(len(card_templates))
    selected=np.zeros(len(card_templates))
    selectedIdx=-1
    prevy=np.inf
    
    for card in range(len(card_templates)):
        found,y,x=findCard(deck_crop,card_templates[card])
        if found:
            oneHotArray[card]=1
            if y<prevy:
                prevy=y
                selectedIdx=card
    
    if prevy<20:
        selected[selectedIdx]=1
    
    return oneHotArray,selected
            