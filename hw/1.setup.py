#IPython is what you are using now to run the notebook
import IPython
print "IPython version:      %6.6s (need at least 1.0)" % IPython.__version__

# Numpy is a library for working with Arrays
import numpy as np
print "Numpy version:        %6.6s (need at least 1.7.1)" % np.__version__

# SciPy implements many different numerical algorithms
import scipy as sp
print "SciPy version:        %6.6s (need at least 0.12.0)" % sp.__version__

# Pandas makes working with data tables easier
import pandas as pd
print "Pandas version:       %6.6s (need at least 0.11.0)" % pd.__version__

# Module for plotting
import matplotlib
print "Mapltolib version:    %6.6s (need at least 1.2.1)" % matplotlib.__version__

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print "Scikit-Learn version: %6.6s (need at least 0.13.1)" % sklearn.__version__

# Requests is a library for getting data from the Web
import requests
print "requests version:     %6.6s (need at least 1.2.3)" % requests.__version__

# Networkx is a library for working with networks
import networkx as nx
print "NetworkX version:     %6.6s (need at least 1.7)" % nx.__version__

#BeautifulSoup is a library to parse HTML and XML documents
import bs4
print "BeautifulSoup version:%6.6s (need at least 4.0)" % bs4.__version__

#MrJob is a library to run map reduce jobs on Amazon's computers
import mrjob
print "Mr Job version:       %6.6s (need at least 0.4)" % mrjob.__version__

#Pattern has lots of tools for working with data from the internet
import pattern
print "Pattern version:      %6.6s (need at least 2.6)" % pattern.__version__

#Seaborn is a nice library for visualizations
import seaborn
print "Seaborn version:      %6.6s (need at least 0.3.1)" % seaborn.__version__




#this line prepares IPython for working with matplotlib
%matplotlib inline  

# this actually imports matplotlib
import matplotlib.pyplot as plt  

x = np.linspace(0, 10, 30)  #array of 30 points from 0 to 10
y = np.sin(x)
z = y + np.random.normal(size=30) * .2
plt.plot(x, y, 'ro-', label='A sine wave')
plt.plot(x, z, 'b-', label='Noisy sine')
plt.legend(loc = 'lower right')
plt.xlabel("X axis")
plt.ylabel("Y axis")   


print "Make a 3 row x 4 column array of random numbers"
x = np.random.random((3, 4))
print x
print

print "Add 1 to every element"
x = x + 1
print x
print

print "Get the element at row 1, column 2"
print x[1, 2]
print

# The colon syntax is called "slicing" the array. 
print "Get the first row"
print x[0, :]
print

print "Get every 2nd column of the first row"
print x[0, ::2]
print



#Three ways to plot binomial distribution
#loop
nx = []
for i in range(500) :
        nx.append(np.random.binomial(500, .5))
#list
nx = [np.random.binomial(500, .5) for i in range(500)]

#just numpy
nx = np.random.binomial(500, .5, size=500)
        
histogram = plt.hist(nx)



####
#Monty Hall Problem
####



"""
Function
--------
simulate_prizedoor

Generate a random array of 0s, 1s, and 2s, representing
hiding a prize between door 0, door 1, and door 2

Parameters
----------
nsim : int
    The number of simulations to run

Returns
-------
sims : array
    Random array of 0s, 1s, and 2s

Example
-------
>>> print simulate_prizedoor(3)
array([0, 0, 2])
"""
def simulate_prizedoor(nsim):
    #compute here
    answer = np.random.randint(0, 3, nsim)
    return answer
#your code here
print simulate_prizedoor(3)



"""
Function
--------
simulate_guess

Return any strategy for guessing which door a prize is behind. This
could be a random strategy, one that always guesses 2, whatever.

Parameters
----------
nsim : int
    The number of simulations to generate guesses for

Returns
-------
guesses : array
    An array of guesses. Each guess is a 0, 1, or 2

Example
-------
>>> print simulate_guess(5)
array([0, 0, 0, 0, 0])
"""

#your code here
def simulate_guess(nsim):
    guesses = np.zeros(nsim, dtype=np.int)
    return guesses
print simulate_guess(5)




"""
Function
--------
goat_door

Simulate the opening of a "goat door" that doesn't contain the prize,
and is different from the contestants guess

Parameters
----------
prizedoors : array
    The door that the prize is behind in each simulation
guesses : array
    THe door that the contestant guessed in each simulation

Returns
-------
goats : array
    The goat door that is opened for each simulation. Each item is 0, 1, or 2, and is different
    from both prizedoors and guesses

Examples
--------
>>> print goat_door(np.array([0, 1, 2]), np.array([1, 1, 1]))
>>> array([2, 2, 0])
"""
#your code here
def goat_door(prizedoors, guesses):
    if len(prizedoors) != len(guesses):
        print "The length of prizedoors and guesses does not match!"
        return None
    goats = []
    
    for i in range(len(prizedoors)):
        goat = [g for g in np.array([0, 1, 2]) if g != prizedoors[i] and g != guesses[i]][0]
        goats.append(goat)
        
    return goats
goats = goat_door(np.array([0, 1, 0]), np.array([1, 1, 1]))
print goats



"""
Function
--------
switch_guess

The strategy that always switches a guess after the goat door is opened

Parameters
----------
guesses : array
     Array of original guesses, for each simulation
goatdoors : array
     Array of revealed goat doors for each simulation

Returns
-------
The new door after switching. Should be different from both guesses and goatdoors

Examples
--------
>>> print switch_guess(np.array([0, 1, 2]), np.array([1, 2, 1]))
>>> array([2, 0, 0])
"""
#your code here
def switch_guess(guesses, goatdoors):
    if len(goatdoors) != len(guesses):
        print "The length of goatdoors and guesses does not match!"
        return None
    newdoors = []
    
    for i in range(len(goatdoors)):
        newdoor = [g for g in np.array([0, 1, 2]) if g != goatdoors[i] and g != guesses[i]][0]
        newdoors.append(newdoor)
        
    return newdoors

print switch_guess(np.array([0, 1, 2]), np.array([1, 2, 1]))

    
"""
Function
--------
win_percentage

Calculate the percent of times that a simulation of guesses is correct

Parameters
-----------
guesses : array
    Guesses for each simulation
prizedoors : array
    Location of prize for each simulation

Returns
--------
percentage : number between 0 and 100
    The win percentage

Examples
---------
>>> print win_percentage(np.array([0, 1, 2]), np.array([0, 0, 0]))
33.333
"""
#your code here

def win_percentage(guesses, prizedoors):
    if len(prizedoors) != len(guesses):
        print "The length of prizedoors and guesses does not match!"
        return None
    wins = 0
    for i in range(len(prizedoors)):
        if guesses[i] == prizedoors[i]:
            wins = wins + 1
    return wins*100.0/len(prizedoors)

    #return 100 * (guesses == prizedoors).mean()

print win_percentage(np.array([0, 1, 2]), np.array([0, 0, 0]))       
guesses = np.array([1,1,0])





#put them together
guesses = simulate_guess(1000)
prizedoors = simulate_prizedoor(1000)
#strategy 1: do not switch
print win_percentage(guesses, prizedoors)
#strategy 2: do switch
goatdoors = goat_door(prizedoors, guesses)
newguesses = switch_guess(guesses, goatdoors)
print win_percentage(newguesses, prizedoors)






