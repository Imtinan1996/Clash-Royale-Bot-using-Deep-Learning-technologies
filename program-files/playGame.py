from screenshot import ScreenGrabber
import cv2
import time
import numpy as np
import random
import pyautogui

from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Activation, Dropout, BatchNormalization, Conv2DTranspose
from keras.models import Model, load_model
from keras.layers.merge import concatenate

def getTiles():

    tile={}

    x_coord=82-26
    y_coord=499-22

    xcounter=1

    for i in range(0,270):
        tile[i]=(int(x_coord),int(y_coord))
        xcounter+=1
        if xcounter>18:
            x_coord=82-26
            y_coord+=21.5
            xcounter=1
        else:
            x_coord+=25.5
            
    return tile

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
            
    deck_crop = frame[560:675,85:415]
    
    #cv2.imwrite("deck.jpg",deck_crop)
    
    oneHotArray=np.zeros(len(card_templates))
    coordinates=[]
    
    for card in range(len(card_templates)):
        found,y,x=findCard(deck_crop,card_templates[card])
        coordinates.append(x)
        if found:
            oneHotArray[card]=1
    
    return oneHotArray,np.array(coordinates)

print("Loading files and models")    
    
eval_deck_templates=np.load("eval_card_templates.npy")
#movePredictor=load_model("moves_predictor.h5")
cardPredictor=load_model("card_predictor.h5")
#positionPredictor=load_model("position_predictor.h5")
tiles=getTiles()

print("Game starting in")
for i in range(10):
    print(10-i)
    time.sleep(1)

cardPos=[(181,904),(276,904),(380,904),(480,904)]    
    
while True:
    
    time.sleep(5)
    
    screen=ScreenGrabber()
    screen=cv2.resize(screen,(418,703))
    board=screen[50:-160,34:-36,:]
    board=cv2.resize(board,(330,460))
    
    board=[board]
    board=np.array(board)
        
    '''
    makeMove=movePredictor.predict(board)
    
    print(makeMove)
    
    makeMove=np.round(makeMove)
    
    if makeMove==1:
    
        print("Move Detected")
    '''    
    deck,coords=cardDetector(screen,eval_deck_templates)
        
    senddeck=[deck]
    senddeck=np.array(senddeck)
    playCard=cardPredictor.predict([board,senddeck])
    
    playCard=playCard[0]
    
    deckProbs=playCard[deck==1]
    deckcoords=coords[deck==1]
    
    print(deckProbs,deckcoords)
    
    chosenCard=np.argmax(deckProbs)
    
    cardx=deckcoords[chosenCard]
    
    deckcoords.sort()
    
    cardPosidx=0
    
    for pos in range(len(deckcoords)):
        if deckcoords[pos]==cardx:
            cardPosidx=pos
    
    print("card on deck: ",cardPosidx)
    
    x,y=cardPos[cardPosidx]
    print("clicking at:",x,y)
    
    pyautogui.click(x,y)
    '''
    position=positionPredictor.predict([board,deck])
    posIdx=np.argmax(position)
    '''
    pos=[(132,540),(418,540)]
    idx=random.random()
    if idx>=0.5:
        idx=1
    else:
        idx=0
    x,y=pos[idx]
    print("clicking at:",x,y)
    pyautogui.click(x,y)
        
        

