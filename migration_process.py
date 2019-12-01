import numpy as np
import matplotlib.pyplot as plt
from math import *
from random import *
from copy import *
from enum import Enum
from transition import *
from parameters import *

"""
    This file stores all the useful functions to simulate the process.
"""

def init_state_matrix(station_size_list, number_of_bikes, size):
	"""
		This function generates randomly an initial state for our process.
	"""

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
	"""
		This function computes a 95% confidence interval giving some data and
		a number of experiences.
	"""

	beta = 1.96
	sigma = np.std(data,axis=0)
	sqrt_n = sqrt(n)

	return (1/sqrt_n)*(beta*sigma)

def compute_mean(data):
	"""
		This function computes the mean of an array of data.
	"""

	m = np.mean(data,axis=0)
	return m

class MigrationProcess:
	"""
		This class is used to describe the Process object.
	"""

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
		"""
			This function prints a MigrationProcess object.
		"""

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

	def station_to_itinerary(self, i, j):
		"""
			This function computes a station to itinerary transition.
		"""

		self.state_matrix[i][i] -= 1
		self.state_matrix[i][j] += 1

	def itinerary_to_station(self, i, j):
		"""
			This function computes an itinerary to station transition.
		"""

		self.state_matrix[i][j] -= 1
		self.state_matrix[j][j] += 1

	def itinerary_to_itinerary(self, i, j, k):
		"""
			This function computes an itinerary to itinerary transition.
		"""

		self.state_matrix[i][j] -= 1
		self.state_matrix[j][k] += 1

	def compute_next_state(self, transition):
		"""
			This function computes the next state according to the type of
			transition.
		"""

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
		"""
			This function simulates an exponential distribution.
		"""

		return np.random.exponential(1/lambda_param)

	def print_rate_matrix(self,rm):
		"""
			This function prints the rate matrix.
		"""

		size = self.number_of_stations
		for i in range(size):
			for j in range(size):
				print(rm[i][j].t_rate,end = ' ')
			print('\n')

	def sta_it_rate(self):
		"""
			This function computes the station to itinerary transition rates.
		"""

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
		"""
			This function computes the itinerary to station (or itinerary)
			transition rates.
		"""

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
		"""
			This function computes the transition rates.
		"""

		size = self.number_of_stations

		sta_it_tr_matrix = np.reshape(self.sta_it_rate(),size*size)
		it_sta_tr_matrix = np.reshape(self.it_sta_rate(),size*size)

		t_rates = np.concatenate((sta_it_tr_matrix,it_sta_tr_matrix))

		return t_rates

	def sum_t_rates(self,t_rates):
		"""
			This function computes the sum of transition rates (q(x,x)).
		"""

		size = self.number_of_stations
		t_rates_sum = sum(o.t_rate for o in t_rates)

		return t_rates_sum

	def compute_weight(self,t_rates,t_rates_sum):
		"""
			This function computes the weight of the transitions.
		"""

		size = self.number_of_stations
		weight_list = np.array([o.t_rate/t_rates_sum for o in t_rates])
		return weight_list

	def select_random_transition(self,weight,t_rates):
		"""
			This function selects a transition randomly according to their
			weight.
		"""

		random_transition_selected = np.random.choice(t_rates, p=weight)
		return random_transition_selected

	def compute_empty_station_time(self, empty_time):
		"""
			This function computes the time each station spend empty.
		"""

		size = self.number_of_stations

		for i in range(size):
			if (self.state_matrix[i][i] == 0):
				self.empty_station_time[i] += empty_time

	def print_progress(self,current,total):
		"""
			This function prints the progress of an operation.
		"""

		b = "Progress : " + str(int((current/total)*100)) + " %"
		print (b, end="\r")

	def estimate_time(N_simulations,T_max,process,theory,print_debug):
		"""
			This function prints the some results about the time each station
			spend empty.
		"""

		k=0
		size = process.number_of_stations
		temporal_emptiness = np.zeros((N_simulations,size))
		spatial_emptiness = np.zeros((N_simulations,size))

		while (k<N_simulations):
			process_copy = deepcopy(process)
			process_copy.simulate_Markov_process(T_max)

			temporal_emptiness[k]=process_copy.empty_station_time
			spatial_emptiness[k]=np.diag(process_copy.state_matrix)==0

			if print_debug:
				process_copy.print_progress(k,N_simulations)
			k+=1

		temporal_emptiness /= (T_max)

		res1=compute_mean(temporal_emptiness)
		int1=compute_conf_interval(temporal_emptiness,N_simulations)
		res2=compute_mean(spatial_emptiness)
		int2=compute_conf_interval(spatial_emptiness,N_simulations)

		if print_debug:
			if theory is None:
				print("%15s %15s %15s %15s %15s" % ("Stations","P Empirique M1","IC 95% M1","P Empirique M2","IC 95% M2"))

				for i in range(0,5):
					print("%15d %15f [-%7f;+%7f] %7f [-%7f;+%7f]" % (i+1,res1[i],int1[i],int1[i],res2[i],int2[i],int2[i]))
			else:
				print("%15s %15s %15s %15s %15s %15s" % ("Stations","P Théorique","P Empirique M1","IC 95% M1","P Empirique M2","IC 95% M2"))

				for i in range(0,5):
					print("%15d %15f %15f [-%7f;+%7f] %7f [-%7f;+%7f]" % (i+1,theory[i],res1[i],int1[i],int1[i],res2[i],int2[i],int2[i]))

				barWidth = 0.2
				y1 = theory.transpose()[0]
				y2 = res1.transpose()
				y3 = res2.transpose()

				r1 = range(0,len(theory))
				r2 = [x + barWidth for x in r1]
				r3 = [x + barWidth*2 for x in r1]

				plt.bar(r1, y1, width = barWidth, color = ['#332288' for i in y1], label = 'Théorique')
				plt.bar(r2, y2, width = barWidth, color = ['#44AA99' for i in y2], label = 'Empirique M1', yerr = int1, ecolor = 'magenta')
				plt.bar(r3, y3, width = barWidth, color = ['#999933' for i in y3], label = 'Empirique M2', yerr = int2, ecolor = 'magenta')
				plt.xticks([r + barWidth*3 / 2 for r in range(len(y1))], [r for r in range(1, len(y1)+1)])
				plt.ylabel("Pourcentage du temps qu'une station est vide sur 150 heures")
				plt.xlabel("Numéro de la station")
				plt.legend()

		return res1, int1



	def simulate_Markov_process(self,T_max):
		"""
			This function simulates our Markov process of Velib.
		"""

		T_current = 0.0

		# while we still have time, we continue to run the process.
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
				transition = self.select_random_transition(weight,t_rates)
				self.compute_next_state(transition)

			#########################################
			## THIRD STEP : we incremente the time ##
			#########################################

			T_current+=T_i

	def run_Markov_process(self, T_max):
		"""
			This function runs our Markov process of Velib.
		"""

		############################
		## GENERATE INITIAL STATE ##
		############################

		print(self.state_matrix)

		self.simulate_Markov_process(T_max)
		self.print_process()

	def coeff_replace(self, j):
		"""
			This function computes some coefficient to add to our theoretical
			model.
		"""

		size = self.number_of_stations
		lsl = self.lambda_station_list
		lim = self.lambda_itinerary_matrix
		rm = self.routing_matrix
		c=1

		for i in range(0,size):
			if(i!=j):
				c += lsl[j]*rm[j][i]/lim[j][i]
		return c


	def compute_theoretical_proba_emptiness(self):
		"""
			This function computes the theoretical probability of emptiness.
		"""

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

		for i in range(0,size):
			for j in range(0,size):
				P[i][j] = (lsl[j] * rm[j][i])/lsl[i]

		P_moins_I = P - I
		P_moins_I[0,:] = first_line

		P_moins_I_inv = np.linalg.inv(P_moins_I)

		alpha_station = np.dot(P_moins_I_inv,b)

		for i in range(0,size):
			for j in range(0,size):
				if i!=j:
					alpha_itinerary[i][j] =  alpha_station[i]*lsl[i]*rm[i][j]/lim[i][j]

		un = np.ones((size,1))
		P_empty = un - alpha_station/(alpha_station.sum()+alpha_itinerary.sum())
		return P_empty


	def bike_number_impact(N_bikes):
		"""
			This function shows the impact of bikes number on stationary
			probability.
		"""

		number_of_stations = 5
		bike_number_impact_matrix = np.zeros((N_bikes+1,number_of_stations))
		bike_number_conf_matrix = np.zeros((N_bikes+1,number_of_stations))

		for i in range(1,N_bikes+1):
			process = MigrationProcess(number_of_stations,
                           i,
                           initial_time,
                           lambda_station_matrix,
                           lambda_station_list_per_mins,
                           lambda_itinerary_matrix,
                           routing_matrix,
                           station_size_list)
			process.print_progress(i,N_bikes)

			res = MigrationProcess.estimate_time(1,T_max,process,None,False)
			bike_number_impact_matrix[i] = res[0]

		barWidth = 0.1
		y1 = bike_number_impact_matrix[:,0]
		y2 = bike_number_impact_matrix[:,1]
		y3 = bike_number_impact_matrix[:,2]
		y4 = bike_number_impact_matrix[:,3]
		y5 = bike_number_impact_matrix[:,4]

		r1 = range(0,len(y1))
		r2 = [x + barWidth for x in r1]
		r3 = [x + barWidth*2 for x in r1]
		r4 = [x + barWidth*3 for x in r1]
		r5 = [x + barWidth*4 for x in r1]

		plt.figure(figsize=(30,5))
		plt.bar(r1, y1, width = barWidth, color = ['#332288' for i in y1], label = 'station 1')
		plt.bar(r2, y2, width = barWidth, color = ['#44AA99' for i in y2], label = 'station 2')
		plt.bar(r3, y3, width = barWidth, color = ['#999933' for i in y3], label = 'station 3')
		plt.bar(r4, y4, width = barWidth, color = ['#CC6677' for i in y4], label = 'station 4')
		plt.bar(r5, y4, width = barWidth, color = ['#AA4499' for i in y5], label = 'station 5')
		plt.xticks([r + barWidth*4 / 2 for r in range(len(y1))], [r for r in range(len(y1))])
		plt.ylabel("Pourcentage du temps qu'une station est vide sur 150 heures")
		plt.xlabel("Nombres total de vélos")
		plt.legend()

	def process_duration_impact(time_limit,theory):
		"""
			This function shows the impact of duration number on stationary
			probability.
		"""

		number_of_stations = 5
		bike_number_impact_matrix = np.zeros((time_limit+1,number_of_stations))
		bike_number_conf_matrix = np.zeros((time_limit+1,number_of_stations))

		for i in range(1,time_limit+1,1):
			T_max = i*60
			process = MigrationProcess(number_of_stations,
                           1,
                           initial_time,
                           lambda_station_matrix,
                           lambda_station_list_per_mins,
                           lambda_itinerary_matrix,
                           routing_matrix,
                           station_size_list)
			process.print_progress(i,time_limit+1)

			res = MigrationProcess.estimate_time(100,T_max,process,None,False)
			bike_number_impact_matrix[i] = abs(res[0] - theory.reshape(1,5)[0])

		barWidth = 0.1

		y1 = bike_number_impact_matrix[:,0]
		y2 = bike_number_impact_matrix[:,1]
		y3 = bike_number_impact_matrix[:,2]
		y4 = bike_number_impact_matrix[:,3]
		y5 = bike_number_impact_matrix[:,4]

		r1 = range(0,len(y1))
		r2 = [x + barWidth for x in r1]
		r3 = [x + barWidth*2 for x in r1]
		r4 = [x + barWidth*3 for x in r1]
		r5 = [x + barWidth*4 for x in r1]

		plt.figure(figsize=(20,5))
		plt.bar(r1, y1, width = barWidth, color = ['#332288' for i in y1], label = 'station 1')
		plt.bar(r2, y2, width = barWidth, color = ['#44AA99' for i in y2], label = 'station 2')
		plt.bar(r3, y3, width = barWidth, color = ['#999933' for i in y3], label = 'station 3')
		plt.bar(r4, y4, width = barWidth, color = ['#CC6677' for i in y4], label = 'station 4')
		plt.bar(r5, y4, width = barWidth, color = ['#AA4499' for i in y5], label = 'station 5')
		plt.xticks([r + barWidth*4 / 2 for r in range(len(y1))], [r for r in range(len(y1))])
		plt.ylabel("Écart à la probabilité stationnaire")
		plt.xlabel("Durée du processus (en heures)")
		plt.legend()
