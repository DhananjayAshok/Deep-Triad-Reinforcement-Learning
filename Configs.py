####
# This file Holds the classes that you will use as your Environment, Game, Action and State Classes

####
from GameSystem.Games import Connect4Game
from GameSystem.States import Connect4State
from GameSystem.Actions import Connect4Action
from GameSystem.Environments import Connect4Environment

GAME_CLASS = Connect4Game
ACTION_CLASS = Connect4Action
ENVIRONMENT_CLASS = Connect4Environment
STATE_CLASS = Connect4State
