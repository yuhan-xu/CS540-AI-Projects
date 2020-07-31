"""
Name: Yuhan Xu
Email: yxu329@wisc.edu
Class: CS 540
Project name: ten_hundred.py
"""


import csv
import math
import numpy as np


"""
 takes in a string with a path to a CSV file formatted as in the link above, and returns the data (without the lat/long 
 columns but retaining all other columns) in a single structure
@:param filepath
@:return return the data (without the lat/long columns but retaining all other columns) in a single structure
"""
def load_data(filepath):
    dict_list = []  # create a list to store dictionaries
    with open(filepath, 'r') as csvfile:  # read file from a filepath
        reader = csv.DictReader(csvfile)  # use csv.DictReader to read a csv file
        # for each row in reader which is a dictionary, append it to the dict_list
        for row in reader:
            dict_list.append(row)
        # for each dictionary d in dict_list, delete lat/long columns
        for d in dict_list:
            del d['Lat']
            del d['Long']

    return dict_list


"""
takes in one row from the data loaded from the previous function, calculates the corresponding x, y values for that 
region as specified in the video, and returns them in a single structure. 
@:param time_series
@:return return corresponding x, y values for that region in a single structure
"""
def calculate_x_y(time_series):
    new_series = list(time_series.values())  # create a list of values of time_series and store in new_series
    del new_series[:2]  # delete the lat/long columns
    new_series = [int(i) for i in new_series]  # change each element in new_series to be int
    t = len(new_series) - 1

    if new_series[t] <= 0:  # if the last value of new_series is smaller or equal to 0, return (nan, nan)
        return math.nan, math.nan

    list1 = []  # create an empty list
    n_div_10 = new_series[t] / 10  # calculate the value of last value of new_series divided by 10
    # iterate through each element in the new_series
    for i in range(len(new_series)):
        # if that specific element is smaller or equal to n/10
        if new_series[i] <= n_div_10:
            list1.append(i)  # append that index to the list1

    if len(list1) == 0:  # if there are not any element in list1
        x = math.nan  # set x to be nan
    else:
        i = max(list1)  # otherwise, find the max of list1 and store it in variable i
        x = t - i  # the final x value is equal to t-i

    list2 = []  # create an empty list called list2
    # find the day with 100 times less cases
    n_div_100 = new_series[t] / 100  # calculate the value of last value of new_series divided by 100
    # iterate through each element in the new_series
    for j in range(len(new_series)):
        if new_series[j] <= n_div_100:  # if that specific element is smaller or equal to n/100
            list2.append(j)  # append that index to the list2

    if len(list2) == 0:  # if there are not any element in list2
        y = math.nan  # set y to be nan
    else:
        j = max(list2)  # otherwise, find the max of list2 and store it in variable j
        y = i - j  # the final y value is equal to i-j

    return x, y  # return the tuple (x,y)

"""
helper method
@:param all_clusters
@:return return distance between points
"""
def dist_between_pts(all_clusters):
    ini_dist = float('inf')

    for i in range(len(all_clusters) - 1):

        for j in range(i+1, len(all_clusters)):
            dis = euclidean_distance(all_clusters[i][1], all_clusters[j][1])

            if dis < ini_dist:
                a = all_clusters[i][0]
                b = all_clusters[j][0]
                ini_dist = dis

    return a, b, ini_dist, (len(all_clusters[i]) + len(all_clusters[j]))


"""
help function to calculate euclidean distance
@:param data1, data2
@:return return euclidean distance
"""
def euclidean_distance(data1, data2):
    distance = math.sqrt((data1[0] - data2[0]) ** 2 + (data1[1] - data2[1]) ** 2)
    return distance

"""
performs single linkage hierarchical agglomerative clustering on the regions with the (x,y) feature representation, and
returns a data structure representing the clustering.
@:param dataset
@:return return a data structure representing the clustering
"""
def hac(dataset):
    filtered = [element for element in dataset if not (math.isnan(element[0]) or math.isnan(element[1]))]
    matrix = []
    index_filtered = []
    clusters = []

    for i in range(len(filtered)):
        index_filtered.append([i, filtered[i]])

    size = len(clusters)

    while len(index_filtered) > 0:
        if size == 0:
            w1, w2, closest, length = dist_between_pts(index_filtered)
            matrix.append([w1, w2, closest, length])
            clusters.append([filtered[w1], filtered[w2], closest, len(filtered)])

    return np.asmatrix(matrix)  # return the numpy matrix

