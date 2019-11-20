import numpy as np
from math import *
from random import *
from copy import *
from enum import Enum
from transition import *


class MigrationProcess:
	def __init__(self,size,number_of_bikes,time,lambda_station_matrix,lambda_itinerary_matrix,routing_matrix,station_size_list):

		self.number_of_stations = size
		self.number_of_bikes = number_of_bikes
		self.time = time
		self.lambda_station_matrix = deepcopy(lambda_station_matrix)
		self.lambda_itinerary_matrix = deepcopy(lambda_itinerary_matrix)
		self.routing_matrix = deepcopy(routing_matrix)
		self.station_size_list = deepcopy(station_size_list)
		self.state_matrix = np.zeros((size,size))
		self.empty_station_time = np.zeros(size)

	def print_process(self):
		print("number of stations : "+str(self.number_of_stations))
		print("number_of_bikes : "+str(self.number_of_bikes))
		print("time : "+str(self.time))
		print("lambda station list : ")
		print(self.lambda_station_matrix)
		print("lambda itinerary matrix : ")
		print(self.lambda_itinerary_matrix)
		print("routing matrix : ")
		print(self.routing_matrix)
		print("station size list : ")
		print(self.station_size_list)
		print("state matrix : ")
		print(self.state_matrix)
		print("empty time per station : ")
		print(self.empty_station_time)

	def get_number_of_stations(self):
		return self.number_of_stations

	def get_lambda_station_matrix(self):
		return self.lambda_station_matrix

	def get_lambda_itinerary_matrix(self):
		return self.lambda_itinerary_matrix

	def get_routing_matrix(self):
		return self.routing_matrix

	def get_station_size_list(self):
		return self.station_size_list

	def get_state_matrix(self):
		return self.state_matrix

	def get_number_of_bikes(self):
		return self.number_of_bikes

	def station_to_itinerary(self, i, j):
	    self.state_matrix[i][i] -= 1
	    self.state_matrix[i][j] += 1

	def itinerary_to_station(self, i, j):
	    self.state_matrix[i][j] -= 1
	    self.state_matrix[j][j] += 1

	def itinerary_to_itinerary(self, i, j, k):
		self.state_matrix[i][j] -= 1
		self.state_matrix[j][k] += 1

	def compute_next_state(self, transition):
		t_type = transition.t_type
		i = transition.t_i
		j = transition.t_j
		k = transition.t_k

		if(t_type == Transition_Type.STATION_TO_ITINERARY):
			self.station_to_itinerary(i, j)
		elif(t_type == Transition_Type.ITINERARY_TO_STATION):
			self.itinerary_to_station(i, j)
		else:
			self.itinerary_to_itinerary(i, j, k)

	def init_state_matrix(self):
		number_of_bikes = self.number_of_bikes
		coord_list = [[i,j] for i in range(5) for j in range(5)]
		while number_of_bikes > 0:
			rand_index = choice(coord_list)
			i = rand_index[0]
			j = rand_index[1]
			if (i==j and self.state_matrix[i][j] == self.station_size_list[i]):
				coord_list.remove(rand_index)
			else:
				self.state_matrix[i][j]+=1
				number_of_bikes -=1

	def exponential_distribution(self,lambda_param):
		return np.random.exponential(1/lambda_param)

	def print_rate_matrix(self,rm):
		size = self.number_of_stations
		for i in range(size):
			for j in range(size):
				print(rm[i][j].t_rate,end = ' ')
			print('\n')

	def sta_it_rate(self):
		size = self.number_of_stations
		m = self.state_matrix
		lsm = self.lambda_station_matrix
		sta_it_rate_array = np.empty((size,size),dtype=object)

		for i in range(size):
			for j in range(size):
				t_i = i
				t_j = j
				t_type = Transition_Type.STATION_TO_ITINERARY
				t_rate = min(m[i][i],1)*lsm[i][j]
				sta_it_rate_array[i][j] = Transition(t_i,t_j,0,t_type,t_rate)

		return sta_it_rate_array

	def it_sta_rate(self):
		size = self.number_of_stations
		m = self.state_matrix
		lim = self.lambda_itinerary_matrix
		ssl = self.station_size_list
		p = self.routing_matrix
		it_sta_rate_array = np.empty((size,size),dtype=object)

		for i in range(size):
			for j in range(size):
				if (m[j][j] >= ssl[j]):
					k = np.random.choice(np.arange(0, size), p=p[j])
					p_j_k = p[j][k]
					t_i = i
					t_j = j
					t_m = k
					t_type = Transition_Type.ITINERARY_TO_ITINERARY
					t_rate = m[i][j] * lim[i][j] * p_j_k
					it_sta_rate_array[i][j] = Transition(t_i,t_j,t_m,t_type,t_rate)
				else:
					t_i = i
					t_j = j
					t_type = Transition_Type.ITINERARY_TO_STATION
					t_rate = m[i][j] * lim[i][j]
					it_sta_rate_array[i][j] = Transition(t_i,t_j,0,t_type,t_rate)

		return it_sta_rate_array

	def compute_transition_rate(self):
		size = self.number_of_stations

		sta_it_tr_matrix = np.reshape(self.sta_it_rate(),size*size)
		it_sta_tr_matrix = np.reshape(self.it_sta_rate(),size*size)

		t_rates = np.concatenate((sta_it_tr_matrix,it_sta_tr_matrix))

		return t_rates

	def sum_t_rates(self,t_rates):
		size = self.number_of_stations
		t_rates_sum = sum(o.t_rate for o in t_rates)

		return t_rates_sum

	def compute_weight(self,t_rates,t_rates_sum):
		size = self.number_of_stations
		weight_list = np.array([o.t_rate/t_rates_sum for o in t_rates])
		return weight_list

	def select_random_transition(self,weight,t_rates):
		random_transition_selected = np.random.choice(t_rates, p=weight)
		return random_transition_selected

	def compute_empty_station_time(self, empty_time):
		size = self.number_of_stations

		for i in range(size):
			if (self.state_matrix[i][i] == 0):
				self.empty_station_time[i] += empty_time


	def estimate_time(N_simulations,T_max,process,size):
		k=0
		current_empty_time = np.zeros(size)
		while (k<N_simulations):
			process_copy = deepcopy(process)
			process_copy.simulate_Markov_process(T_max)
			current_empty_time+=process_copy.empty_station_time
			# process_copy.print_process()
			print(k)
			k+=1
		current_empty_time /= (T_max*N_simulations)
		print(current_empty_time)


	def simulate_Markov_process(self,T_max):

		T_current = 0.0

		# while we still have time, we continue to run the process
		while (T_current < T_max):

			###############################################################
			## FIRST STEP : compute t_i with an exponential distribution ##
			###############################################################

			t_rates = self.compute_transition_rate()
			t_rates_sum = self.sum_t_rates(t_rates)

			T_i = self.exponential_distribution(t_rates_sum)

			self.compute_empty_station_time(min(T_i,T_max-T_current))

			if(T_i+T_current <= T_max):
			##########################################
			## SECOND STEP : compute the next state ##
			##########################################

				weight = self.compute_weight(t_rates,t_rates_sum)
				# print([str(o.t_i)+","+str(o.t_j) for o in t_rates])
				# print(weight)
				transition = self.select_random_transition(weight,t_rates)
				# transition.print_transition()
				# print(self.state_matrix)
				self.compute_next_state(transition)
				# print(self.state_matrix)

			#########################################
			## THIRD STEP : we incremente the time ##
			#########################################

			T_current+=T_i

	def run_Markov_process(self, T_max):

		############################
		## GENERATE INITIAL STATE ##
		############################

		# self.init_state_matrix()
		self.state_matrix = np.array(
		    [   [20  ,1  ,0  ,0  ,0],
			    [1  ,16  ,1  ,0  ,0],
			    [0  ,1  ,17  ,1  ,0],
			    [0  ,0  ,1  ,13  ,1],
			    [0  ,0  ,0  ,1  ,18]])

		self.simulate_Markov_process(T_max)