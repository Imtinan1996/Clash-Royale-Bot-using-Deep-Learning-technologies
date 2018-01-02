from detectCards import cardDetector
from detectClock import findClock
import cv2
import numpy as np
import os

videoDirectory='gameplay_footage/'
footage = [ file for file in os.listdir(videoDirectory)]

clock = cv2.imread("clock-small.png")
card_templates=np.load("card_templates.npy")

print (len(footage)," gameplay footages found in library")

moves_cnn_data=[]                       #pd.DataFrame(columns=['image','playedCard'])
clock_pos_data=[]                       #pd.DataFrame(columns=['image','cardPlayed','positionPlayed'])
card_played_data=[]                     #pd.DataFrame(columns=['image','availableDeck','cardPlayed'])

mfile_counter=1
clfile_counter=1
cfile_counter=1

for gameplay in range(len(footage)):
        
   
    print("Processing gameplay: ",gameplay+1,"/",len(footage))
    
    video=cv2.VideoCapture(videoDirectory+footage[gameplay])
    videoRunning=True
    
    prevFrameDeck=[]
    prevFrameImg=[]
    prevFrameCardSelected=[]
    
    while videoRunning:
        
        videoRunning,frame=video.read()
        
        if videoRunning is False:
            break
        
        frame=cv2.resize(frame,(418,703))
        
        gameBoard=frame[140:-103,46:-42,:]
        
        moveMade,movePos=findClock(frame,clock)
        currentDeck,selectedCard=cardDetector(frame,card_templates)
        
        if moveMade:
        #    moves_cnn_data.append([gameBoard,1])
            clock_pos_data.append([prevFrameImg,prevFrameCardSelected,movePos])
            card_played_data.append([prevFrameImg,prevFrameDeck,prevFrameCardSelected])
        #else:
        #    moves_cnn_data.append([gameBoard,0])
        '''
        if len(moves_cnn_data)>=1000:
            np.save("datafiles/moves_cnn_data/moves-file-"+str(mfile_counter),moves_cnn_data)
            mfile_counter+=1
            del moves_cnn_data[:]
            moves_cnn_data=[] 
        '''    
        if len(clock_pos_data)>=1000:
            np.save("datafiles/clock_pos_data/clockpos-file-"+str(clfile_counter),clock_pos_data)
            clfile_counter+=1
            del clock_pos_data[:]
            clock_pos_data=[] 
        
        if len(card_played_data)>=1000:
            np.save("datafiles/card_played_data/cardplayed-file-"+str(cfile_counter),card_played_data)
            cfile_counter+=1
            del card_played_data[:]
            card_played_data=[] 
        
        prevFrameDeck=currentDeck
        prevFrameImg=gameBoard
        prevFrameCardSelected=selectedCard
        

#np.save("datafiles/moves_cnn_data/moves-file-"+str(mfile_counter),moves_cnn_data)
np.save("datafiles/clock_pos_data/clockpos-file-"+str(clfile_counter),clock_pos_data)
np.save("datafiles/card_played_data/cardplayed-file-"+str(cfile_counter),card_played_data)

