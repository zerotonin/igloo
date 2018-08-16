# -*- coding: utf-8 -*-
"""
Created on Tue May 17 10:59:42 2016

@author: bgeurten

This class holds  functions and look up tables needed to calculate the walking 
velocity and bout duration of a Drosophila depending of the animals temperature 
and the temperature during its ontogenisis. Therefore only four distinct 
simulations can be run: 1) Drosophila adults reared at 18°C 2) Drosophila
adults reared at 25°C 3) Drosophila adults reared at 30°C 4) Drosophila larva
reared at 25°C

For perfomance and smoothness of the trajectory it is recommended to use the
locomotionInterpolation class

"""
import numpy as np

def pickDataSet(rearTemp,probVec):
    '''
    This function picks the correct dataset for the rearing temperature.
    
    @param rearTemp mixed can be int 18,25,30 or string 'larval'
    @param probVec  list of list containing the data
    @return velocity and durationlists with the data, will be empty if rearTemp
            is not as expected.
    '''
    if (rearTemp == 18):
        velocity = probVec[2,0]
        duration = probVec[3,0]
    elif (rearTemp == 25):
        velocity = probVec[2,1]
        duration = probVec[3,1]
    elif (rearTemp == 30):
        velocity = probVec[2,2]
        duration = probVec[3,2]
    elif (rearTemp == 'larval'):
        velocity = probVec[2,3]
        duration = probVec[3,3]
    else:
        velocity = []
        duration = []
    return velocity,duration

def findValue(t,p,dataSet):
    '''
    This is the central function that finds the closest combination of 
    original data and the momentary values in the simulation. 
    
    @param t       float Drosophila body temperature
    @param p       float random number between 0 and 1
    @param dataset list with either the original velocity or duration values
    '''
    # get correct temperature subset
    subSet = dataSet[dataSet[:,0] == (round(t)),1:3]
    # get the absolute difference of all data pValues to our random p
    pVals =  np.abs(subSet[:,0]-p)   
    # the minimum of this vector should be the p value closest to our random p
    values =subSet[np.where(pVals == pVals.min()),1] 
    values = values.ravel()
    #ä return values and use -1 to take the last
    return values[-1]