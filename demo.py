# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:51:01 2016

@author: bgeurten
"""

#import scipy.io as sio
import numpy as np
import IGLOO as mod
import matplotlib.pyplot as plt
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


#  __    __     ______     _____     ______     __        
# /\ "-./  \   /\  __ \   /\  __-.  /\  ___\   /\ \       
# \ \ \-./\ \  \ \ \/\ \  \ \ \/\ \ \ \  __\   \ \ \____  
#  \ \_\ \ \_\  \ \_____\  \ \____-  \ \_____\  \ \_____\ 
#   \/_/  \/_/   \/_____/   \/____/   \/_____/   \/_____/ 

# built simulation object with the following parameters:
# gradientExt    : Our gradient starts at 10°C and ends at 35°C
# walkDur        : The duration of the simulation is set to 200 seconds
# startPos       : The simulated animal starts 62.5 mm from the cold end (-1 activates random start position)
# rearingT       : The simulated animal is reared at 25°C
# simulationType : There are two different simulation types, here we use the interpolation mode
# gradientDist   : The simulated gradient is 75 mm long

# using the upper variables to create the simulation environment
modObj = mod.IGLOO(gradientExt=(10,35),walkDur=200.0,startPos=62.5,rearingT=25.0,
                  simulationType='interpolate',gradientDist=75.0)



#  ______     __     __   __     ______     __         ______        ______   __         __  __    
# /\  ___\   /\ \   /\ "-.\ \   /\  ___\   /\ \       /\  ___\      /\  ___\ /\ \       /\ \_\ \   
# \ \___  \  \ \ \  \ \ \-.  \  \ \ \__ \  \ \ \____  \ \  __\      \ \  __\ \ \ \____  \ \____ \  
#  \/\_____\  \ \_\  \ \_\\"\_\  \ \_____\  \ \_____\  \ \_____\     \ \_\    \ \_____\  \/\_____\ 
#   \/_____/   \/_/   \/_/ \/_/   \/_____/   \/_____/   \/_____/      \/_/     \/_____/   \/_____/ 
                                                                                                 

# now we simulate a single fly
modObj.simulateSingleFly()
# info to the user that the script will only go on if the figure is closed
print('Routine will continue as soon as you close the figure!')
#plot this fly | showNow is a boolean flag that is needed to see the figure directly
modObj.plotSingleTrace(showNow=True)
# save single trajectory to a txt file
modObj.save4TXTSingleTra('/home/bgeurten/test.txt')


#  ______   __         __  __        ______   ______     ______   __  __     __         ______     ______   __     ______     __   __    
# /\  ___\ /\ \       /\ \_\ \      /\  == \ /\  __ \   /\  == \ /\ \/\ \   /\ \       /\  __ \   /\__  _\ /\ \   /\  __ \   /\ "-.\ \   
# \ \  __\ \ \ \____  \ \____ \     \ \  _-/ \ \ \/\ \  \ \  _-/ \ \ \_\ \  \ \ \____  \ \  __ \  \/_/\ \/ \ \ \  \ \ \/\ \  \ \ \-.  \  
#  \ \_\    \ \_____\  \/\_____\     \ \_\    \ \_____\  \ \_\    \ \_____\  \ \_____\  \ \_\ \_\    \ \_\  \ \_\  \ \_____\  \ \_\\"\_\ 
#   \/_/     \/_____/   \/_____/      \/_/     \/_____/   \/_/     \/_____/   \/_____/   \/_/\/_/     \/_/   \/_/   \/_____/   \/_/ \/_/ 
                                                                                                                                       

# We simulate 202 flies first argument is the number of flies, second if you want to have the
# single trajectories plotted
modObj.simulateFlyPopulation(202,True)
# calculate the histogram of the ambient temperature (2)  
# 1: position 2: ambient temperature 3: body temperature
modObj.calcHistogram(2)
# info to the user that the script will only end if all figures are closed
print('Routine will end as soon as you close the figures!')
#plot the histogram of the ambient temperature
modObj.plotHistogram()
# save all trajectories to text files beginning with the prefix 'testTrace' in the directory '/home/bgeurten/testSave'
modObj.save4TXTPopulation('/home/bgeurten/testSave','testTrace')
#saving the histogram data
modObj.save4TXTHistogram('/home/bgeurten/testHist.txt')