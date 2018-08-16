# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:35:26 2016

@author: bgeurten

This class holds the mathematical functions needed for interpolating the walking 
velocity and bout duration of a Drosophila depending of the animals temperature 
and the temperature during its ontogenisis. Luckily the data can be fully fitted
using Gaussian functions.
The modelcan also operate on the original data see locomotionOnRawData


"""

import numpy as np

    
def gauss(x,a,x0,sigma):    
    """
    Implementation of a Gaussian distribution
        
                           2
                 (x - x0)
              - ------------
                           2
                (2.0*sigma)
    f(x) = a*e 
    
    @param x     float x - value(s as list)   
    @param a     float factor before the gaussian 
    @param x0    float mu of the distribution
    @param sigma float sigma of the distribution
    @return list of floats with the results corresponding to the x-values
    
    """
    return a*np.exp(-np.power((x-x0)/(2.*sigma), 2.))

def ratio(x,a,b):   
    """
    Implementation of a simple ratio function
    
             a
    f(x) = -----
           x + b    
       
    @param x     float x - value(s as list)   
    @param a     float see above 
    @param b     float see above 
    @return list of floats with the results corresponding to the x-values
    
    """

    return a/(x+b)

def linear(x,a,b):
    """
    Implementation of a simple linear function
    
    f(x) = a x + b
       
    @param x     float x - value(s as list)   
    @param a     float see above -4.71*x^4+167.91*x^3-2568.55*x^2+9292.99*x+0.05
    @param b     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return a*x+b
    
def poly2(x,a,b,c):       
    """
    Implementation of a 2nd degree polynom
        
              2
    f(x) = a x  + b x + c
       
    @param x     float x - value(s as list)   
    @param a     float see above 
    @param b     float see above 
    @param c     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return a*x**2+b*x+c
    
def poly3(x,a,b,c,d):    
    """
    Implementation of a 3rd degree polynom
        
              3      2
    f(x) = a x  + b x  + c x + d
       
    @param x     float x - value(s as list)   
    @param a     float see above 
    @param b     float see above 
    @param c     float see above 
    @param d     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return a*x**3+b*x**2+c*x+d
    
def poly4(x,a,b,c,d,e):
    
    """
    Implementation of a 4th degree polynom
    
              4      3      2
    f(x) = a x  + b x  + c x  + d x + e
       
    @param x     float x - value(s as list)   
    @param a     float see above 
    @param b     float see above 
    @param c     float see above 
    @param d     float see above 
    @param e     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return e*x**4+a*x**3+b*x**2+c*x+d
    
def hillUp(x,o,n,s,b):

    """
    Implementation of the rising slope of a hill equation
    
                n
               x
    f(x) = b ------- + s
              n    n
             o  + x
       
    @param x     float x - value(s as list)   
    @param o     float see above 
    @param n     float see above 
    @param s     float see above 
    @param b     float see above 
    @return list of floats with the results corresponding to the x-values
    """    
    return  b*(x**n/(o**n +x**n))+s

def hillDown(x,o,n,s,b):
    """
    Implementation of the falling  slope of a hill equation
    
               b
    f(x) = -------- + s2
                  n
               /x\
           1 + |-|
               \o/

       
    @param x     float x - value(s as list)   
    @param o     float see above 
    @param n     float see above 
    @param s     float see above 
    @param b     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return (b/(1+(x/o)**n))+s

def hillDouble(x,o,n,s,s2,b):
    """
    Implementation of a Hill function
    
           /     n       \
           |    x        | /    b        \
    f(x) = |b ------- + s| |-------- + s2|
           |   n    n    | |       n     |
           \  o  + x     / |    /x\      |
                           |1 + |-|      |
                           \    \o/      /
       
    @param x     float x - value(s as list)   
    @param o     float see above 
    @param n     float see above 
    @param s     float see above 
    @param s2    float see above 
    @param b     float see above 
    @return list of floats with the results corresponding to the x-values
    """
    return hillUp(x,o,n,s,b)*hillDown(x,o,n,s2,b)
    
    
def velFunc(x_p,x_t,v):    
    """
    This function calculates the fit of the velocity in dependens of the 
    temperature and rearing temperature of the ADULT animal. The function 
    consitits of two gaussian fits for the temperature domain and a 2nd
    degree polynom for the p Values. This 2nd degree polyom changes with the
    rearing temperature.
    
    @param x_p float random value between  0 and 1
    @param x_t float Drosophila body temperature
    @param v   list of floats containing the rearing temperature depending 
               parameters
    @return list of floats with new velocity values
    """
    res = (gauss(x_t,1.6,34.0,2.27)+gauss(x_t,1.8,26.0,5.41))*poly2(x_p,v[0],v[1],v[2])  
    return res  

    
    
def durFunc(x_p,x_t,d):    
    """
    This function calculates the fit of the fit duration in dependence of the 
    temperature and rearing temperature of the ADULT animal.  The function 
    consitits of two gaussian fits for the temperature domain and a 2nd
    degree polynom for the p Values. In this case both the temperature domain 
    and the p-values are dependent on the rearing temperature.
    
    @param x_p float random value between  0 and 1
    @param x_t float Drosophila body temperature
    @param d   list of floats containing the rearing temperature depending 
               parameters
    @return list of floats with new step duration values
    """
    
    res  = (gauss(x_t,d[0],34.0,d[1])+gauss(x_t,d[2],18,d[3])) * poly3(x_p,d[4],d[5],d[6],d[7]) 
    return res
    
def velFuncLarva(x_p,x_t):    
    """
    This function calculates the fit of the velocity in dependens of the 
    temperature of the LARVAL animal (reared at 18°C). The function 
    consitits of a 4th degree polynom for the temperature domain and a 3rd
    degree polynom for the p Values. 
    
    @param x_p float random value between  0 and 1
    @param x_t float Drosophila body temperature
    @return list of floats with new velocity values
    
                    4           3            2
    f(x) = (-4.71) x  + 167.91 x  - 2568.55 x  + 9292.99 x + 0.05
    """
    res = poly4(x_t,-4.71341,167.91083,-2568.54657,9292.98836,0.04707)*poly3(x_p,-0.00180,0.00225,-0.00106,0.000009)     
    return res  

    
    
def durFuncLarva(x_p,x_t):    
    """
    This function calculates the fit of the step duration in dependens of the 
    temperature of the LARVAL animal (reared at 18°C). The function 
    consitits of two Gaussian fits for the temperature domain and two Gaussian
    fits for the p Values. 
    
    @param x_p float random value between  0 and 1
    @param x_t float Drosophila body temperature
    @return list of floats with new step duration values
    """
    res = (gauss(x_t,8.62,9.0,0.82)*gauss(x_p,11.74,0.69,-0.12))+(gauss(x_t,1860.53,34.0,6.79)*gauss(x_p,1860.53,4.96,-0.56)) 
    return res
    
def calcParameters(T):
    """
    This function returns the rearing temperature depending parameters to 
    adjust the velocity and step duration functions. Most parameters can be fit
    with a 2nd degree polynom. As only 3 different rearing temperatures were 
    used the fits might not be to reliable and it is saver to stay with the 
    three measured rearing temperatures: 18,25,30 °C
    
    @param T float rearing temperature
    @return lists two lists 1) v velocity parameters 2) d duration parameters
    """
    if (T == 'larval'):
        v = []
        d = []
    else:
    
        v =np.empty(3)
        v[0]    = poly2(T, 0.12099083,     -5.66690323,      70.7646241)
        v[1]    = poly2(T,-3.74142886e-02,  2.11954454e+00,  -4.34662354e+01)
        v[2]    = poly2(T,-0.07461945,      3.14576967,     -22.92563255)
        
        
        d =np.empty(8)
        d[0]  = linear(T, 0.12916312,  -2.812052 )
        d[1]  = poly2( T, 0.14534352,  -7.95321438, 110.60810592)
        d[2]  = poly2( T, 0.02662904,  -1.28284224,  15.65186953)
        d[3]  = poly2( T,11.66468748,-641.51376808,8753.61384674)
        d[4]  = poly2( T, 0.14445866,  -6.92317723,  75.43250874)
        d[5]  = poly2( T,-0.24815044,  11.88971369,-129.60720025)
        d[6]  = poly2( T ,0.15978689,  -7.64871786,  83.63873179)
        d[7]  = poly2( T,-0.0612502,    2.92865988, -32.21078329)
        
    return v,d