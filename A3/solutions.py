# solutions.py
# ------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

'''Implement the methods from the classes in inference.py here'''

import util
from util import raiseNotDefined, manhattanDistance
#from distanceCalculator import manhattanDistance
import random
import busters
import inference

def normalize(self):
    """
    Normalize the distribution such that the total value of all keys sums
    to 1. The ratio of values for all keys will remain the same. In the case
    where the total value of the distribution is 0, do nothing.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> dist.normalize()
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
    >>> dist['e'] = 4
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
    >>> empty = DiscreteDistribution()
    >>> empty.normalize()
    >>> empty
    {}
    """
    # Used for normalizing
    totalValue = self.total()
    itemsDict = self.items()

    # Items are not empty 
    if itemsDict is not {}: 
        for i, val in self.items():
            if val == 0: 
                # Cannot be normalized, so return 
                self[i] = 0.0
            elif val != 0: 
                # Normalize 
                self[i] = val/totalValue

def sample(self):
    """
    Draw a random sample from the distribution and return the i, weighted
    by the values associated with each i.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> N = 100 000.0
    >>> samples = [dist.sample() for _ in range(int(N))]
    >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
    0.2
    >>> round(samples.count('b') * 1.0/N, 1)
    0.4
    >>> round(samples.count('c') * 1.0/N, 1)
    0.4
    >>> round(samples.count('d') * 1.0/N, 1)
    0.0
    """
    # List to choose return value from
    samples = []

    # Used for denormalizing
    totalValue = self.total()
    itemsDict = self.items()

    # Check if each item is denormalized
    for item, vals in itemsDict:
        if vals >= 1: 
            for i in range(vals):
                # Append if denormalized
                samples.append(item)
        elif 0 < vals < 1: 
            # Denormalize if already normalized
            denormalized = totalValue * vals
    
            for j in range(denormalized): 
                # Append when denormalized
                samples.append(item)

    # Output variable
    randSample = samples[random.randint(0, len(samples)-1)]
    
    return randSample

def getObservationProb(self, noisyDistance, pman_pos, ghostPosition, jail_pos):
    """
    Return the probability P(noisyDistance | pman_pos, ghostPosition).
    """
    "*** YOUR CODE HERE ***"
    # handling the special case
    if ghostPosition == jail_pos:
        # if the ghost is captured in the jail position, then the observation is None with probability 1
        if noisyDistance == None:
            return 1
        # is distance reading is not None, ghost in jail with probability 0
        elif noisyDistance != None:
            return 0

    elif ghostPosition != jail_pos: # if the ghost is running free on the map
            # else we calculate the actual distance between Pacman and the ghost then return value
        if noisyDistance == None:
            return 0

    #If none of these, 
    else:
        pacman_to_ghost = manhattanDistance(pman_pos, ghostPosition)
        return busters.getObservationProbability(noisyDistance, pacman_to_ghost)


def observeUpdate(self, observation, gameState):
    """
    Update beliefs based on the distance observation and Pacman's position.

    The observation is the noisy Manhattan distance to the ghost you are
    tracking.

    self.allPositions is a list of the possible ghost positions, including
    the jail position. You should only consider positions that are in
    self.allPositions.

    The update model is not entirely stationary: it may depend on Pacman's
    current position. However, this is not a problem, as Pacman's current
    position is known.
    """
    "*** YOUR CODE HERE ***"
    #Get pacman position
    pman_pos = gameState.getPacmanPosition()

    #Get jail position
    jail_pos = self.getJailPosition()

    #Getting distribution
    dist = inference.DiscreteDistribution()

    #iterate over all the positions
    for i in range(0, len(self.allPositions)):

        #set the current position 
        pos = self.allPositions[i]

        #get current probabilities
        probability = self.getObservationProb(observation, pman_pos, pos, jail_pos)

        #set probability distribution
        dist[pos] = self.beliefs[pos] * probability
    
    #normalize
    dist.normalize()
    self.beliefs = dist
    

def elapseTime(self, gameState):
    """
    Predict beliefs in response to a time step passing from the current
    state.

    The transition model is not entirely stationary: it may depend on
    Pacman's current position. However, this is not a problem, as Pacman's
    current position is known.
    """
    "*** YOUR CODE HERE ***"
    pos_dict = inference.DiscreteDistribution()
    for oldPos in self.allPositions:
        # for each old position p, the new position ditribution is the probablity
        # that the ghost is at position p at time t + 1, given ghost at oldPos at time t
        newPosDist = self.getPositionDistribution(gameState, oldPos)

        # updating all position
        for pos in self.allPositions:
            pos_dict[pos] = pos_dict[pos] + newPosDist[pos] * self.beliefs[oldPos]

    # update the Discrete Distribution to what the updated positions looks like
    self.beliefs = pos_dict