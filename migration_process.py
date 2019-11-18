import numpy as np
from random import *
from copy import *

class MigrationProcess:
    def __init__(self,size,number_of_bikes,time,lambda_station_list,lambda_itinerary_matrix,routing_matrix,station_size_list):

    	self.number_of_stations = size
    	self.number_of_bikes = number_of_bikes
    	self.time = time
    	self.lambda_station_list = deepcopy(lambda_station_list)
    	self.lambda_itinerary_matrix = deepcopy(lambda_itinerary_matrix)
    	self.routing_matrix = deepcopy(routing_matrix)
    	self.station_size_list = deepcopy(station_size_list)
    	self.state_matrix = np.zeros((size,size))

    def print_process(self):
    	print("number of stations : "+str(self.number_of_stations))
    	print("number_of_bikes : "+str(self.number_of_bikes))
    	print("time : "+str(self.time))
    	print("lambda station list : ")
    	print(self.lambda_station_list)
    	print("lambda itinerary matrix : ")
    	print(self.lambda_itinerary_matrix)
    	print("routing matrix : ")
    	print(self.routing_matrix)
    	print("station size list : ")
    	print(self.station_size_list)
    	print("state matrix : ")
    	print(self.state_matrix)

    def get_number_of_stations(self):
    	return self.number_of_stations

    def get_lambda_station_list(self):
    	return self.lambda_station_list

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