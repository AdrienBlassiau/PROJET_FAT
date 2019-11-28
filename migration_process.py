import numpy as np
from math import *
from random import *
from copy import *
from enum import Enum
from transition import *

def init_state_matrix(station_size_list, number_of_bikes, size):
	state_matrix = np.zeros((size,size))

	coord_list = [[i,j] for i in range(5) for j in range(5)]
	while number_of_bikes > 0:
		rand_index = choice(coord_list)
		i = rand_index[0]
		j = rand_index[1]
		if (i==j and state_matrix[i][j] >= station_size_list[i]):
			coord_list.remove(rand_index)
		else:
			state_matrix[i][j] +=1
			number_of_bikes -=1

	return state_matrix

def compute_conf_interval(data,n):
	beta = 1.96
	sigma = np.std(data,axis=0)
	sqrt_n = sqrt(n)

	return (1/sqrt_n)*(beta*sigma)

def compute_mean(data):
	m = np.mean(data,axis=0)
	return m

class MigrationProcess:
	def __init__(self,size,number_of_bikes,time,lambda_station_matrix,lambda_station_list,lambda_itinerary_matrix,routing_matrix,station_size_list):

		self.number_of_stations = size
		self.number_of_bikes = number_of_bikes
		self.time = time
		self.lambda_station_matrix = deepcopy(lambda_station_matrix)
		self.lambda_station_list = deepcopy(lambda_station_list)
		self.lambda_itinerary_matrix = deepcopy(lambda_itinerary_matrix)
		self.routing_matrix = deepcopy(routing_matrix)
		self.station_size_list = deepcopy(station_size_list)
		self.state_matrix = init_state_matrix(station_size_list, number_of_bikes, size)
		self.empty_station_time = np.zeros(size)

	def print_process(self):
		print("Nombre de stations : "+str(self.number_of_stations))
		print("Nombre de vélos dans le réseau : "+str(self.number_of_bikes))
		print("Temps initial : "+str(self.time))
		print("Intensité des occurences de demande de vélos à la station i (1/min) : ")
		print(self.lambda_station_list)
		print("Intensité des trajets en vélo de la station i à la station j (1/min) : ")
		print(self.lambda_itinerary_matrix)
		print("Matrice de routage : ")
		print(self.routing_matrix)
		print("Nombre de places par station : ")
		print(self.station_size_list)
		print("État courant (nombre de vélos par colonie), généré aléatoirement à l'initialisation: ")
		print(self.state_matrix)
		print("Temps durant lequel chaque station est vide (nul au démarrage ...) : ")
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

	def print_progress(self,current,total):
		b = "Progress : " + str(int((current/total)*100)) + " %"
		print (b, end="\r")

	def estimate_time(N_simulations,T_max,process,size):
		k=0
		temporal_emptiness = np.zeros((N_simulations,size))
		spatial_emptiness = np.zeros((N_simulations,size))

		while (k<N_simulations):
			process_copy = deepcopy(process)
			process_copy.simulate_Markov_process(T_max)

			temporal_emptiness[k]=process_copy.empty_station_time
			spatial_emptiness[k]=np.diag(process_copy.state_matrix)==0

			process_copy.print_progress(k,N_simulations)
			k+=1

		temporal_emptiness /= (T_max)

		res1=compute_mean(temporal_emptiness)
		int1=compute_conf_interval(temporal_emptiness,N_simulations)
		res2=compute_mean(spatial_emptiness)
		int2=compute_conf_interval(spatial_emptiness,N_simulations)

		print("%15s %15s %15s %15s %15s" % ("stations","Méthode 1","Écart 1","Méthode 2","Écart 2"))

		for i in range(0,5):
			print("%15f %15f %15f %15f %15f" % (i+1,res1[i],int1[i],res2[i],int2[i]))


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

		print(self.state_matrix)

		self.simulate_Markov_process(T_max)
		self.print_process()

	def coeff_replace(self, j):
		size = self.number_of_stations
		lsl = self.lambda_station_list
		lim = self.lambda_itinerary_matrix
		rm = self.routing_matrix
		c=1

		for i in range(0,size):
			if(i!=j):
				c += lsl[j]*rm[j][i]/lim[j][i]
		return c


	def compute_alpha(self):
		size = self.number_of_stations
		lsl = self.lambda_station_list
		lim = self.lambda_itinerary_matrix
		rm = self.routing_matrix
		P = np.zeros((size,size))
		alpha_itinerary = np.zeros((size,size))
		I = np.eye(size)
		first_line = np.ones(size)
		b = np.zeros((size,1))
		b[0] = 1
		for k in range(0,size):
			first_line[k] = self.coeff_replace(k)
		# print(first_line)

		for i in range(0,size):
			for j in range(0,size):
				P[i][j] = (lsl[j] * rm[j][i])/lsl[i]

		P_moins_I = P - I
		P_moins_I[0,:] = first_line

		P_moins_I_inv = np.linalg.inv(P_moins_I)

		alpha_station = np.dot(P_moins_I_inv,b)
		# print(sum(alpha_station))
		# print(alpha_station)

		for i in range(0,size):
			for j in range(0,size):
				if i!=j:
					alpha_itinerary[i][j] =  alpha_station[i]*lsl[i]*rm[i][j]/lim[i][j]

		# print(alpha_itinerary)
		# print("sum : ",alpha_station.sum()+alpha_itinerary.sum())
		un = np.ones((size,1))
		P_empty = un - alpha_station/(alpha_station.sum()+alpha_itinerary.sum())
		return P_empty