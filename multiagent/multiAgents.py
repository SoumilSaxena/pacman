# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        minimum = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])
        dir = currentGameState.getPacmanState().getDirection()
        Difference_Scores = successorGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearest = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        new1 = [manhattanDistance(newPos, food) for food in newFood.asList()]
        new2 = 0 if not new1 else min(new1)

        temp = nearest - new2

        if minimum <= 1 or action == Directions.STOP:
            return 0
        if Difference_Scores > 0:
            return 8
        elif temp > 0:
            return 4
        elif action == dir:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        numberOfGhosts = gameState.getNumAgents() - 1
        def Minimum(gameState, dep, Index):

            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            MinimumVal = 999999

            Next = gameState.getLegalActions(Index)
            for action in Next:
                NextInLine = gameState.generateSuccessor(Index, action)
                if Index == (gameState.getNumAgents() - 1):
                    MinimumVal = min(MinimumVal, Maximum(NextInLine, dep))
                else:
                    MinimumVal = min(MinimumVal, Minimum(NextInLine, dep, Index + 1))
            return MinimumVal

        def Maximum(gameState, dep):
            current  = dep + 1
            MaximumVal = -999999
            if gameState.isWin() or gameState.isLose() or current == self.depth:  # Terminal Test
                return self.evaluationFunction(gameState)

            Next = gameState.getLegalActions(0)
            for action in Next:
                successor = gameState.generateSuccessor(0, action)
                MaximumVal = max(MaximumVal, Minimum(successor, current, 1))
            return MaximumVal

        def minmax():
            Next = gameState.getLegalActions(0)
            currentScore = -999999
            back = ''
            for action in Next:
                temp = gameState.generateSuccessor(0, action)
                curr = Minimum(temp, 0, 1)
                if curr > currentScore:
                    back = action
                    currentScore = curr
            return back
        return minmax()
       #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def AlphaBeta(gameState, ag, dep, x, y):
            ReturnList = []
            if not gameState.getLegalActions(ag):
                return self.evaluationFunction(gameState), 0
            if dep == self.depth:
                return self.evaluationFunction(gameState), 0
            if ag == gameState.getNumAgents() - 1:
                dep += 1
            if ag == gameState.getNumAgents() - 1:
                Succesor_agent = self.index
            else:
                Succesor_agent = ag + 1
            for action in gameState.getLegalActions(ag):
                if not ReturnList:
                    Succesor_Val = AlphaBeta(gameState.generateSuccessor(ag, action), Succesor_agent, dep, x, y)

                    ReturnList.append(Succesor_Val[0])
                    ReturnList.append(action)

                    if ag == self.index:
                        x = max(ReturnList[0], x)
                    else:
                        y = min(ReturnList[0], y)
                else:
                    if ReturnList[0] > y and ag == self.index:
                        return ReturnList

                    if ReturnList[0] < x and ag != self.index:
                        return ReturnList

                    prev = ReturnList[0]
                    Succesor_Val = AlphaBeta(gameState.generateSuccessor(ag, action), Succesor_agent, dep, x, y)
                    if ag == self.index:
                        if Succesor_Val[0] > prev:
                            ReturnList[0] = Succesor_Val[0]
                            ReturnList[1] = action
                            x = max(ReturnList[0], x)
                    else:
                        if Succesor_Val[0] < prev:
                            ReturnList[0] = Succesor_Val[0]
                            ReturnList[1] = action
                            y = min(ReturnList[0], y)
            return ReturnList
        return AlphaBeta(gameState, self.index, 0, -99999, 99999)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxValue(state, Index, depth):
            Index = 0
            legal = state.getLegalActions(Index)

            if not legal or depth == self.depth:
                return self.evaluationFunction(state)

            maxValue = max(Expected(state.generateSuccessor(Index, action), \
                                    Index + 1, depth + 1) for action in legal)

            return maxValue
        def Expected(state, Index, depth):
            counter = gameState.getNumAgents()
            legal = state.getLegalActions(Index)
            if not legal:
                return self.evaluationFunction(state)

            e = 0
            probabilty = 1.0 / len(legal)
            for action in legal:
                if Index == counter - 1:
                    currentExpValue = maxValue(state.generateSuccessor(Index, action), \
                                               Index, depth)
                else:
                    currentExpValue = Expected(state.generateSuccessor(Index, action), \
                                               Index + 1, depth)
                e += currentExpValue * probabilty

            return e



        actions = gameState.getLegalActions(0)
        SETofActions = {}
        for action in actions:
            SETofActions[action] = Expected(gameState.generateSuccessor(0, action), 1, 1)
        return max(SETofActions, key=SETofActions.get)

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: In this case we have to evaluate states. Made 3 functions score of ghostscore, capsule eater (ghosteater) and food eater. we added score of current game state to the return values of the three functions.
    """
    "*** YOUR CODE HERE ***"

    def Food(State):
        distance = []
        for food in State.getFood().asList():
            distance.append(1.0 / manhattanDistance(State.getPacmanPosition(), food))
        if len(distance) > 0:
            return max(distance)
        else:
            return 0
    def EatGhost(State):
        counter = []
        for x in State.getCapsules():
            counter.append(50.0 / manhattanDistance(State.getPacmanPosition(), x))
        if len(counter) > 0:
            return max(counter)
        else:
            return 0
    def GhostScore(State):
        score = 0
        for ghost in State.getGhostStates():
            distance = manhattanDistance(State.getPacmanPosition(), ghost.getPosition())
            if ghost.scaredTimer > 0:
                score += pow(max(8 - distance, 0), 2)
            else:
                score -= pow(max(7 - distance, 0), 2)
        return score
    return GhostScore(currentGameState) + Food(currentGameState) + EatGhost(currentGameState) + currentGameState.getScore()

    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction