import numpy as np
from math import *
from random import *
from copy import *
from enum import Enum

class Transition_Type(Enum):
	STATION_TO_ITINERARY = 1
	ITINERARY_TO_STATION = 2
	ITINERARY_TO_ITINERARY = 3

class Transition:
	def __init__(self,t_i,t_j,t_k,t_type,t_rate):
		self.t_i = t_i
		self.t_j = t_j
		self.t_k = t_k
		self.t_type = t_type
		self.t_rate = t_rate

	def print_transition(self):
		print("t_i : "+str(self.t_i))
		print("t_j : "+str(self.t_j))
		print("t_k : "+str(self.t_k))
		print("t_type : "+str(self.t_type))
		print("t_rate : "+str(self.t_rate))