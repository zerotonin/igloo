# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:42:55 2016

@author: bgeurten
"""
from scipy import stats
from tqdm import tqdm
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Subfunctions 
import locomotionInterpolation as lI
import locomotionOnRawData as lORD

class IGLOO:
    def __init__(self,startPos=25.,gradientExt=(12.,32.),gradientDist=50.,
                 walkDur=300.,rearingT=25.,sps=50,simulationType ='interpolate'):      
        """
        This function intialises the monte carlo random walk class. Here most 
        of the simulation task will be done. Each fly is represented by 3 values 
        its position [mm], the environment temperature at this position [°C] and
        its body temperature [°C]. The gradient is a one dimensional strip of
        x mm length that has a linear temperature gradient with gradientExt as
        the gradient extreme temperatures. The lower extreme temperature is 
        situated at 0 mm the hotter extreme temperature at the far end of the 
        gradient. The fly is further described by ots rearing temperature 
        (default: 25°C) and its preferred temperature (default: 21°C). The 
        original null model is a random walk.   
        
        @param startPos         float  default: 25          
            start position of the fly in mm on the gradient 
        @param gradientExt      tuple  default: (12.,32.)
            temperature extremes of the linear gradient in °C
        @param gradiantDist     float  default: 50.0
            total length of the gradient in mm
        @param walkDur          float  default: 300.0
            total duration of the simulation in seconds
        @param rearingT         float  default: 25.0
            rearing temperature of the fly in °C if you simulate on the 
            original data only 18.0, 25.0 and 30.0 can be used. If instead of
            float rearingT is set as 'larval' larvae reared at 25°C are 
            simulated.
        @param sps              int    default: 50
            Samples per second. After the walking duration is reached. The whole
            trajectory will be resampled with this framerate.
        @param simulationType   string default: 'interpolate'
            String that defines if the simulation is run on the original data 
            set or on the interpolation functions. The later is much faster and
            allows to set the rearing temperature to any value between 18 and 
            30. To run on the original data set this string to "onData"
        """
        # set values to inputs or defaults
        self.startPosition  = startPos          # start position of the animal [mm]
        self.position       = startPos          # current position of the animal [mm]
        self.gradientExt    = gradientExt       # cold and hot max values for the gradient [°C]
        self.gradientDist   = gradientDist      # length of the gradient [mm]
        self.sps            = sps               # samples per second       
        self.walkDur        = walkDur           # duration of the simulation [s]
        self.rearingT       = rearingT          # temperature it which the animal was reared [°C]
        self.simulationType = simulationType    # either 'onData' or 'interpolation'
        
        #where is this file on the disk? Needed to load original test data
        HERE = os.path.dirname(os.path.abspath(__file__))
        
        # calculate from defaults and inputs        
        self.ambientT       = np.mean(self.gradientExt)     #temperature of the gradient at the current position of the animal [°C]
        self.drosoT         = self.ambientT     # current body temperature of the animal set to ambient for start conditions
        self.degPerMM       = abs(np.diff(self.gradientExt))/self.gradientDist
        self.degPerMM       = self.degPerMM[0]  # factor with which to calculate from current position to current ambient temperature
        
        # empty variables
        self.vParams,self.dParams,self.time,self.direction,self.step= np.zeros((5))
        self.histData       = (0,0)
        self.selfTestData   = (0,0)
        self.flyPop         = np.zeros((int(np.ceil(self.walkDur*self.sps)),4))

        # on data simulation variables
        if(self.simulationType == 'onData'):
            mat_contents = sio.loadmat(os.path.join(HERE, 'probVecs.mat'))
            self.allData = mat_contents['probVecs']
            self.velData,self.durData = lORD.pickDataSet(self.rearingT,self.allData )
        else:
            self.allData = []
            self.velData = []
            self.durData = []
        
        # reset temp trace
        self.resetTempTrace()
        
    def resetTempTrace(self):
        """
        This function resets the fly and gradient so that a new simulation trial
        can begin, e.g. fly returns to original position and default temperature
        time is reset to zero.
        
        The results are saved in self.position .drosoT .time and .tempTrace
        
        self.tempTrace is a nx4 vector, where column 1) time [s], 2) position
        [mm], 3) ambient temperature [°C], 4) body temperature [°C]
        """
        # reset start position
        if (self.startPosition == -1):
             self.position  = self.gradientDist*rd.random()
        else:
            self.position  = self.startPosition
        # update the ambience temperature to that position
        self.updateAmbientTemp()
        # set the body temperature to the ambience temperature for adaptation
        self.drosoT    = self.ambientT
        # set time to zero
        self.time      = 0.0
        # initialise the temperature trace
        self.tempTrace = np.array([self.time,self.position,self.ambientT, self.drosoT])
        
    

    def stepFunc(self):    
        """
        This function creates a step during random walk. The direction is 
        randomly determined. Velocity and step duration are determined by the 
        subroutine self.move()
        """
        
        # if not do a random walk
        self.direction = rd.choice([-1,1])
        self.move()
            
                    
                
      
    def move(self):   
        """
        This is a subroutine of the step function. This function updates the 
        fly position. This can be done using the original data (simulation: 
        'onData') or interpolation functions (simulation: 'interpolated').
        
        The results are saved in self.step and self.time
        """
        # make two random numbers between 0 and 1
        pValueVelocity = rd.random()
        pValueDuration = rd.random()
        duration = 0
        velocity = 0
        # interpolate the velocity and duration based on rearing temperature 
        # and body temperature
        if (self.simulationType=='interpolate'):
            # check for larval simulation, as no parameters are needed than
            if (self.rearingT == 'larval'):
                velocity  = lI.velFuncLarva(pValueVelocity,self.drosoT)
                duration  = lI.durFuncLarva(pValueDuration,self.drosoT)
                
            else:
                velocity  = lI.velFunc(pValueVelocity,self.drosoT,self.vParams)
                duration  = lI.durFunc(pValueDuration,self.drosoT,self.dParams)
        elif(self.simulationType=='onData'):
            velocity  = lORD.findValue(self.drosoT,pValueVelocity,self.velData)
            duration  = lORD.findValue(self.drosoT,pValueDuration,self.durData)
        #Add the duration of the step to the clock
        self.time = self.time + duration
        # and the step to the position
        self.step = self.direction*duration*velocity
        
    def updatePosition(self):    
        """
        This function updates the position of the animal. This is important, as
        the ends of the gradient are reflective, e.g.: The gradient is 50 mm 
        long the fly is at 47 mm and makes a 10 mm step towards the near end. 
        In this case the animal walk 3 mm to the near end and gets reflected 
        for 7 mm. The new fly position would be 43 mm.  
        
        The result is saved in self.position
        """
        # update the new position
        newPos       = self.position + self.step
        
        # check if the new position is  outside the gradient, if this is the
        # case reflect the position from the wall
        gradientDist = self.gradientDist
        if (newPos > gradientDist):
            overshoot = newPos - gradientDist
            newPos = gradientDist-overshoot
        elif (newPos < 0.0):
            newPos = abs(newPos)
        self.position = newPos 
        
    def updateAmbientTemp(self):    
        """
        This subroutine updates the ambient temperature to the spot where the 
        fly arrived after the position was updated. This is trivial.
        
        
        The result is saved in self.ambientT
        """
        # calculate new ambience temperature
        newAmbientT = self.position*self.degPerMM + self.gradientExt[0]
        self.ambientT = newAmbientT
    
    def drosoTbyConduction(self):
        """
        This function calculates the temperature change for animals in our TLM
        model. It uses Newtons law for convection and models the Drosophila as an
        3 by 0.5 mm cylinder walking in a 3  mm wide cylinder. Internally the
        values given will be transferred to the correct dimensions. The
        following values are set conductance: 
        air 0.0262 W/(m²*K)  David R. Lide (Hrsg.): CRC Handbook of Chemistry and Physics. 90. Auflage. (Internet-Version: 2010), CRC Press/Taylor and Francis, Boca Raton, FL, Fluid Properties, S. 6-184. Werte gelten bei 300 K.
        water 0.6 W/(m²*K)   https://de.wikipedia.org/wiki/Eigenschaften_des_Wassers#W.C3.A4rmeleitf.C3.A4higkeit 
         
        The formula for rate of heat flow is:      dQ               T1-T2
                                              Q  = -- = lambda * A* -----
                                                   dt                 D
                                                   
        where D is the wallt to wall thickness of the object lambda is its 
        conductance A its surface T1, T2 the temperature of the object and its
        surroundings.
           
        Drosophila surface'  as a cylinder would be for the cylinders hull: 
        2*pi*r*l and the two disks: (2*pi*r**2)*2. If the cylinder has an 
        r = 0.5mm and a length of l =2mm the resulting surface is 7.85 mm²
        
        To calculate the temperature change we have to know the conductance of 
        our fly which we approximate as a cylinder of water lambda 0.6. A 
        cylinder of that size would have a mass of  1.57 mg    
        
        Model limits: Drosophila is a 1 times 2 mm cylinder consisting of  
        1.58 mg of water suspended in air. 

        The result is saved in self.drosoT
        """ 
        #physical size of the conductors
        conductance = 0.6 #np.array([0.37,  0.0262])
        surface     = 7.85
        
        #time spent in the new environment
        if (len(self.tempTrace.shape) >1):    # for all after the first iteration
            t = self.time -self.tempTrace[-2,0]
        else: # first iteration
            t = self.time
        # convert to correct dimension 

        # The formula for rate of heat flow is:      dQ               T1-T2
        #                                       Q  = -- = lambda * A* -----
        #                                            d                  D
        # where D is the wallt to wall thickness of the object lambda is its 
        # conductance A its surface T1, T2 the temperature of the object and its
        # surroundings.
        # C to K because the Celsius and Kalvin have the same relative units
        
        dT= self.ambientT - self.drosoT  
        # surface mm² to m²
        surface = surface / 10.0**6.0 
        # mm to m
        D = 1.0 / 10.0**3.0
        
        
        # Calculate the heat flow        
        Q = conductance*surface*(dT/D)*t
        
        # Q has the dimension Watt*secvParamsond = Joule
        
        # Joule = energy needed to change 1 gr of waters temperature by 0.239 K
        # Drosophila weighs approx. 0.25mg and consists mainly out of water and
        # therefore 1 Joule should heat up a 1.57 mg mass of water by 152.23 °K
        temperature_change  = Q*152.23
        
        # if our temperature change overshoots it will actually be limited to 
        # the ambient temperature
        if (self.ambientT > self.drosoT) and (self.drosoT+temperature_change> self.ambientT):
            self.drosoT = self.ambientT
        elif (self.ambientT < self.drosoT) and (self.drosoT+temperature_change < self.ambientT):
            self.drosoT = self.ambientT
        else:
            self.drosoT = self.drosoT + temperature_change
        


                        
    def simulateSingleFly(self):    
        """
        This function simulates the random walk of a single fly. For the
        duration given by self.walkDur in [s] on the gradient defined 
        by self.gradientExt and self.gradientDist
        To simulate more than one fly, please use simulateFlyPopulation
        
        The result is saved in self.tempTrace
        
        self.tempTrace is a nx4 vector, where column 1) time [s], 2) position
        [mm], 3) ambient temperature [°C], 4) body temperature [°C] and n is 
        self.walkDur*self.sps
        """
        # reset single trace variable
        self.resetTempTrace()
        #calculate rearing temperature dependent function paramteres
        self.vParams,self.dParams = lI.calcParameters(self.rearingT)
        # simulate random walk until the duration is exceeded
        while (self.time < self.walkDur):
            # make a step
            self.stepFunc()
            # update fly position and ambient temperature
            self.updatePosition()
            self.updateAmbientTemp()
            # change Drosophilas temperature by conduction
            self.drosoTbyConduction()
            # add sample to vector
            self.tempTrace = np.vstack((self.tempTrace,np.array([self.time,self.position,self.ambientT, self.drosoT])))
                  
            
        #interpolate the data to fix samplerate
        xi = np.linspace(0,self.walkDur,self.walkDur*self.sps) # time vector
        pos = np.interp(xi,self.tempTrace[:,0],self.tempTrace[:,1]) # positions
        aT = np.interp(xi,self.tempTrace[:,0],self.tempTrace[:,2]) # ambient Temperature
        dT = np.interp(xi,self.tempTrace[:,0],self.tempTrace[:,3]) # animal Temperature
            
        #built return matrix
        self.tempTrace = np.column_stack((xi,pos))
        self.tempTrace = np.column_stack((self.tempTrace,aT))
        self.tempTrace = np.column_stack((self.tempTrace,dT))
            
    def simulateFlyPopulation(self,flyN,plotFlag = False):    
        """
        This function wraps the simualteSingleFly function and itereates it for
        the number of flies given by flyN.
        
        @param flyN     int  number of flies to be simulated       
        @param plotFlag bool default: 0     if set to 1 all trajectories are 
               plotted
        
        The result is saved in self.flypop
        """
        # preallocate output variable
        self.flyPop = np.zeros((int(self.walkDur*self.sps),3,flyN))
        # simulate all flys
        for flyI in tqdm(xrange(0,flyN)):
            self.simulateSingleFly()
            self.flyPop[:,:,flyI] = self.tempTrace[:,1:4]
            # plot if needed
            if (plotFlag == True):
                self.plotSingleTrace(1)
        self.flyPop=(self.tempTrace[:,0], self.flyPop)
                              
    def calcHistogram(self,tempFlag):    
        """
        This function calculates a  position histogram for every fly in the 
        population and normalises it to its surface. Afterwards it calculates 
        mean +/- SEM of all histograms and normalises again. 
        
        The result is saved in self.histData a tupel consisting of the 
        number of samples normalised to their total and the bins of the 
        histogram. The number of samples variable has two rows 1st is mean and 
        the 2nd is the SEM of each bin
        
        @param tempFlag int if set to 3 it calculates ambient temperature if 
                            set to 4 it calculates body temperature
        """   
        
        # number of bins in the histogran
        binNum =self.gradientExt[1]-self.gradientExt[0]
        #shortHand
        flyPop = self.flyPop[1]
        # number of animals simulated
        animalNum = flyPop.shape[2]        
        # bin centers
        bins = np.linspace(self.gradientExt[0]-0.5,self.gradientExt[1]+0.5,binNum+2)
        # preallocation of return values         
        histAT = np.zeros((animalNum,binNum+1))
        
        for flyI in range(0,animalNum-1):
            histAT[flyI,:],binedges = np.histogram(flyPop[:,tempFlag,flyI],bins=bins,density=True)
        self.bins =bins
        self.histData = (np.mean(histAT,axis=0), stats.sem(histAT, axis=0))
        self.histData = (self.histData[0]/sum(self.histData[0]),self.histData[1])
        
           
        
        
##################################################################################
#  ______   __         ______     ______   ______   __     __   __     ______    #
# /\  == \ /\ \       /\  __ \   /\__  _\ /\__  _\ /\ \   /\ "-.\ \   /\  ___\   #
# \ \  _-/ \ \ \____  \ \ \/\ \  \/_/\ \/ \/_/\ \/ \ \ \  \ \ \-.  \  \ \ \__ \  #
#  \ \_\    \ \_____\  \ \_____\    \ \_\    \ \_\  \ \_\  \ \_\\"\_\  \ \_____\ #
#   \/_/     \/_____/   \/_____/     \/_/     \/_/   \/_/   \/_/ \/_/   \/_____/ #
##################################################################################  
                              
    def plotSingleTrace(self,fHandle = -1,showNow=False):    
        """
        This plots the last calculated trace as line plots. There are 3 
        subplots 1) Time vs Position 2) Time vs Ambient Temperature 3) Time
        vs Body Temperature
        
        @param fHandle  int default: -1 figure handle if set to default a new 
                        figure will be opend. Otherwise the figure with this 
                        handle will be cleared and the data plotted there.
        """    
        if (fHandle == -1):
            plt.figure()
        else:
            plt.figure(fHandle)
    
        plt.subplot(311)
        plt.plot(self.tempTrace[:,0],self.tempTrace[:,1])
        plt.xlabel('time [s]')
        plt.ylabel('gradientPosition [mm]')
        plt.title('Animal Position')
        
        plt.subplot(312)
        plt.plot(self.tempTrace[:,0],self.tempTrace[:,2])
        plt.xlabel('time [s]')
        plt.ylabel('temperature [deg C]')
        plt.title('Ambient Temperature')
        
        plt.subplot(313)
        plt.plot(self.tempTrace[:,0],self.tempTrace[:,3])
        plt.xlabel('time [s]')
        plt.ylabel('animal temperature [deg C]')
        plt.title('Animal temperature')
        plt.draw()

        if showNow == True:
            plt.show()
        
    def plotHistogram(self,fHandle = -1):    
        """
        This function calculates a temperature histogram and plots it as bar
        plots to the figure handle provided by the user
        
        @param fHandle  int default: -1 figure handle if set to default a new 
                        figure will be opend. Otherwise the figure with this 
                        handle will be cleared and the data plotted there.
        """    
        
        #open figure
        if (fHandle == -1):
            plt.figure()
        else:
            plt.figure(fHandle)
            
        # number of bins in the histogran
        binNum =self.gradientExt[1]-self.gradientExt[0]
        
        index = np.linspace(self.gradientExt[0],self.gradientExt[1],binNum+1)
        bar_width =  0.35

        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        
        plt.bar(index, self.histData[0], bar_width,
                alpha=opacity,
                color='b',
                yerr=self.histData[1],
                error_kw=error_config,
                label='animal T ')
        
    
        plt.xlabel('temperature [deg C]')
        plt.ylabel('percentage')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        

        
###############################################################################
#           ______   __     __         ______        __     ______            #
#          /\  ___\ /\ \   /\ \       /\  ___\      /\ \   /\  __ \           #
#          \ \  __\ \ \ \  \ \ \____  \ \  __\      \ \ \  \ \ \/\ \          #
#           \ \_\    \ \_\  \ \_____\  \ \_____\     \ \_\  \ \_____\         #
#            \/_/     \/_/   \/_____/   \/_____/      \/_/   \/_____/         #                         
###############################################################################        

    def save4MatlabSingleTra(self,fPos):
        """
        This function saves the following variables to a Matlab file:
        
        @return lastTra is a nx4 vector, where column 1) time [s], 
                2) position [mm], 3) ambient temperature [°C], 4) body temperature 
                [°C] and n is self.walkDur*self.sps       
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.
        """    
        sio.savemat(fPos,  {'lastTra':self.tempTrace,
                            'startPos':self.startPosition,
                            'gradientExt':self.gradientExt,
                            'gradientDist':self.gradientDist,
                            'sps':self.sps,
                            'walkDur':self.walkDur ,
                            'rearingT':self.rearingT,
                            'simulationType':self.simulationType})
                            
        
    def save4TXTSingleTra(self,fpos,trace=''):
        """
        This function saves the following variables to a txt file:
        
        @return lastTra is a nx4 vector, where column 1) time [s], 
                2) position [mm], 3) ambient temperature [°C], 4) body temperature 
                [°C] and n is self.walkDur*self.sps.   in a 4.5 float format      
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.
        """ 
        # if no trace is defined take the last trace in memory
        if len(trace) == 0:
            trace = self.tempTrace
        # make file head information including simulation type etc. 
        headStr = self.make_txt_header()
        # write to disk
        np.savetxt(fpos,trace,fmt='%4.5f',header= headStr)
    
    def save4TXTPopulation(self,dirName,prefix='IGLOO'):
        """
        This function saves the following variables for each fly into a single
        ASCII text file with 4 digits before and 5 after the point. :
        
        @return lastTra is a nx4 vector, where column 1) time [s], 
                2) position [mm], 3) ambient temperature [°C], 4) body temperature 
                [°C] and n is self.walkDur*self.sps.   in a 4.5 float format      
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  dirName string the absolute position of the director you want to 
                save your single fly trace to.
        @param  prefix string with the prefix for all fly trajectory files. Example:
                if set to 'cantonS' and you simulated one hundred flies the files 
                will be called cantonS_0000.txt to cantonS_0099.txt default: 'IGLOO'
        """    
        #short cut to variable
        flyPop = self.flyPop[1]
        # number of animals simulated
        animalNum = flyPop.shape[2]
        # main loop in which the filename is generated and the data saved        
        for flyI in range(0,animalNum):
            #make file name
            fName = prefix + '_' + str(flyI).zfill(4)
            fPos  = os.path.join(dirName, fName)
            # for populations of flies the allways identical time column is not saved with 
            # the rest of the data in memory. So now it has to be added before writing
            # everything to disk
            self.save4TXTSingleTra(fPos,trace= np.column_stack((self.tempTrace[:,0],flyPop[:,:,flyI])))


    def save4MatlabPopulation(self,fPos):
        """
        This function saves the following variables to a Matlab file:
        
        @return time a vetor of self.walkDur*self.sps length holding the time 
                in seconds
        @return flyPop a list of vectors nx4 vector, where column
                1) position [mm], 2) ambient temperature [°C], 3) body temperature 
                [°C] and n is self.walkDur*self.sps. Each fly is saved in a single 
                entry of the list
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.
        """    
        sio.savemat(fPos,  {'time':self.flyPop[0],
                            'flyPop':self.flyPop[1],
                            'startPos':self.startPosition ,
                            'gradientExt':self.gradientExt ,
                            'gradientDist':self.gradientDist,
                            'sps':self.sps,
                            'walkDur':self.walkDur,
                            'rearingT':self.rearingT,
                            'simulationType':self.simulationType})
       
        
    def save4TXTHistogram(self,fPos):
        '''
        This function saves the histogram data to disk. The resulting file has three
        columns: 
        1) The middle temperature of each bin
        2) The mean of the histograms of all flies / normalised to an integral of one
        3) The standard error of the mean of the histograms of all flies
        
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.

        '''
        # get middle of bins for histograms
        index          = np.linspace(self.gradientExt[0],self.gradientExt[1],len(self.bins)-1)
        
        #make nx3 matrix consiting of bins, mean and sem
        histReturnData = np.row_stack((index,self.histData[0]))
        histReturnData = np.row_stack((histReturnData,self.histData[1]))
        histReturnData = histReturnData.T

        # save environmental data such as start position and simulation type
        headStr = self.make_txt_header()
        # add further information
        headStr = headStr + '\nFirst column is temperature bin. Second column is mean and third column is the SEM.\n' 

        # write to disk 
        np.savetxt(fPos,histReturnData,fmt='%4.5f',header= headStr)

    def save4MatlabHistogram(self,fPos):
        """
        This function saves the following variables to a Matlab file:
        
        @return histData a tupel consisting of the number of samples
                normalised to their total and the bins of the histogram. The  
                number of samples variable has two rows 1st is mean and the 2nd 
                is the SEM of each bin       
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.
        """    
        sio.savemat(fPos,  {'histData':self.histData,
                            'startPos':self.startPosition ,
                            'gradientExt':self.gradientExt ,
                            'gradientDist':self.gradientDist,
                            'sps':self.sps ,
                            'walkDur':self.walkDur ,
                            'rearingT':self.rearingT,
                            'simulationType':self.simulationType})
        pass
    
    def save4MatlabAll(self,fPos):
        """
        This function saves the following variables to a Matlab file:
        
        @return lastTra is a nx4 vector, where column 1) time [s], 2) position
                [mm], 3) ambient temperature [°C], 4) body temperature [°C] and 
                n is self.walkDur*self.sps  
        @return histData a tupel consisting of the number of samples
                normalised to their total and the bins of the histogram. The  
                number of samples variable has two rows 1st is mean and the 2nd 
                is the SEM of each bin         
        @return time a vetor of self.walkDur*self.sps length holding the time 
                in seconds
        @return flyPop a list of vectors nx3 vector, where column 1) position
                [mm], 2) ambient temperature [°C], 3) body temperature [°C] and 
                n is self.walkDur*self.sps. Each fly is saved in a single entry
                of the list  
        @return startPos start position of the fly in mm on the gradient 
        @return gradientExt temperature extremes of the linear gradient in °C
        @return gradiantDist total length of the gradient in mm
        @return sps Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        @return walkDur total duration of the simulation in seconds
        @return rearingT  rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                 25°C are simulated.
        @return simulationType String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
                
        @param  fPos string the absolute position of the file you want to save
                data to.
        """    
        sio.savemat(fPos,  {'lastTra':self.tempTrace,
                            'histData':self.histData,
                            'time':self.flyPop[0],
                            'flyPop':self.flyPop[1],
                            'startPos':self.startPosition ,
                            'gradientExt':self.gradientExt ,
                            'gradientDist':self.gradientDist,
                            'walkDur':self.walkDur ,
                            'rearingT':self.rearingT,
                            'simulationType':self.simulationType})
    
    def make_txt_header(self):
        """
        All our txt output files have to record the environmental data of the simulation
        as shown below. These are put in a single string variable and saved as the header
        of the text file.

        startPos 
                start position of the fly in mm on the gradient 
        gradientExt 
                temperature extremes of the linear gradient in °C
        gradiantDist 
                total length of the gradient in mm
        sps 
                Samples per second. After the walking duration is reached. 
                The wholetrajectory will be resampled with this framerate.
        walkDur 
                total duration of the simulation in seconds
        rearingT  
                rearing temperature of the fly in °C if you simulate 
                on the original data only 18.0, 25.0 and 30.0 can be used. If
                instead of float rearingT is set as 'larval' larvae reared at
                25°C are simulated.
        simulationType 
                String that defines if the simulation is run on 
                the original data set or on the interpolation functions. The 
                later is much faster and allows to set the rearing temperature 
                to any value between 18 and 30. To run on the original data set 
                this string to "onData"
        
        @return headStr a string carrying the above information and written into all
                text files.
        """ 
        headStr = ' __     ______     __         ______     ______      \n' 
        headStr = headStr + '/\ \   /\  ___\   /\ \       /\  __ \   /\  __ \     \n' 
        headStr = headStr + '\ \ \  \ \ \__ \  \ \ \____  \ \ \/\ \  \ \ \/\ \    \n' 
        headStr = headStr + ' \ \_\  \ \_____\  \ \_____\  \ \_____\  \ \_____\   \n' 
        headStr = headStr + '  \/_/   \/_____/   \/_____/   \/_____/   \/_____/   \n\n' 
        headStr = headStr + 'gradient distance   [mm]: ' + str(self.gradientDist) + '\n'
        headStr = headStr + 'gradient extremes   [°C]: ' + str(self.gradientExt[0]) + ' ' + str(self.gradientExt[1]) + '\n'
        headStr = headStr + 'start position      [mm]: ' + str(self.startPosition) + '\n'
        headStr = headStr + 'simulation duration  [s]: ' + str(self.walkDur) + '\n'
        headStr = headStr + 'rearing temperature [°C]: ' + str(self.rearingT) + '\n'
        headStr = headStr + 'samples per second      : ' + str(self.sps) + '\n'
        headStr = headStr + 'simulation type         : ' + str(self.simulationType) + '\n'
        return headStr
        
        