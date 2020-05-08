from Opponents.Opponents import HumanOpponent, HyperionOpponent, RandomOpponent
from Configs import ACTION_CLASS, GAME_CLASS, ENVIRONMENT_CLASS


g = ENVIRONMENT_CLASS()
h = HumanOpponent()
r = RandomOpponent(blocking=True, winning=True)
hyp = HyperionOpponent()

g.reset(hyp, r)
