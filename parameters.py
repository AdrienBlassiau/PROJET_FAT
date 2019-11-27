import numpy as np

####################################
######## INITIAL PARAMETERS ########
####################################

# We define our routing matrix
routing_matrix = np.array(
    [[0.0,  0.2,  0.3,  0.2,  0.3],
     [0.2,  0.0,  0.3,  0.2,  0.3],
     [0.2,  0.25, 0.0,  0.25, 0.3],
     [0.15, 0.2,  0.3,  0.0,  0.35],
     [0.2,  0.25, 0.35, 0.2,  0.0]])
# routing_matrix = np.array([
#     [0      ,0.22    ,0.32    ,0.2     ,0.26],
#     [0.17   ,0       ,0.34    ,0.21    ,0.28],
#     [0.19   ,0.26    ,0       ,0.24    ,0.31],
#     [0.17   ,0.22    ,0.33    ,0       ,0.28],
#     [0.18   ,0.24    ,0.35    ,0.23    ,0]])

# We define our lambda_i coefficients
lambda_station_list_per_hours = np.array([2.8,  3.7,  5.5,  3.5,  4.6])

lambda_station_list_per_mins = lambda_station_list_per_hours / 60

lambda_station_matrix = lambda_station_list_per_mins.reshape(5,1) * routing_matrix

# We define our lambda_ij coefficients
# We put 1 one first diagonal to avoid dividing by zero, it will not be used
travel_time_itinerary_matrix = np.array(
    [[1.0,  3.0,  5.0,  7.0,  7.0],
     [2.0,  1.0,  2.0,  5.0,  5.0],
     [4.0,  2.0,  1.0,  3.0,  3.0],
     [8.0,  6.0,  4.0,  1.0,  2.0],
     [7.0,  7.0,  5.0,  2.0,  1.0]])

lambda_itinerary_matrix = 1/travel_time_itinerary_matrix
np.fill_diagonal(lambda_itinerary_matrix, 0)

# We define the size of our stations v_i
station_size_list = np.array([24,  20,  20,  15,  20])

number_of_stations = 5
number_of_bikes = 80
initial_time = 0.0
T_max = 150*60