# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:11:22 2021

@author: Steven
"""
import numpy as np
import sys
#import matching
#import best_matching_score

direct_connection_dict = {}
#connections based on floorplan
direct_connection_dict['i'] = ['II']
direct_connection_dict['II'] = ['i','iii','A','E','F', '1','5','6']
direct_connection_dict['iii'] = ['II','I','L','9','12'] # G, F, 6 en 7 mogen hier mogelijks ook bij mr dan moeten ze ook geupdate worden
direct_connection_dict['A'] = ['II','B']
direct_connection_dict['B'] = ['A','C','D','E']
direct_connection_dict['C'] = ['B','D']
direct_connection_dict['D'] = ['B','C','E','G','H']
direct_connection_dict['E'] = ['II','B','D','G']
direct_connection_dict['F'] = ['II','G','I']
direct_connection_dict['G'] = ['D','E','F','H','I']
direct_connection_dict['H'] = ['D','G','M']
direct_connection_dict['I'] = ['iii','F','G'] # M ook of niet?
direct_connection_dict['J'] = ['I','K']
direct_connection_dict['K'] = ['J','L']
direct_connection_dict['L'] = ['iii','S','12','19']  # wat met de trappenhal
direct_connection_dict['M'] = ['H','N','P','Q'] #Q valt te betwisten
direct_connection_dict['N'] = ['O','P']
direct_connection_dict['O'] = ['N','P']
direct_connection_dict['P'] = ['M','N','O','Q','R','S']
direct_connection_dict['Q'] = ['M','P','R','S']
direct_connection_dict['R'] = ['P','Q','S']
direct_connection_dict['S'] = ['P','Q','R','19'] # moet 'iii' hier ook bij?
direct_connection_dict['1'] = ['II','2']
direct_connection_dict['2'] = ['1','3','4','5']
direct_connection_dict['3'] = ['2']
direct_connection_dict['4'] = ['2','5','7']
direct_connection_dict['5'] = ['II','2','4','7']
direct_connection_dict['6'] = ['II','7','9']
direct_connection_dict['7'] = ['4','5','6','8','9']
direct_connection_dict['8'] = ['4','7','13']
direct_connection_dict['9'] = ['iii','6','7','10']
direct_connection_dict['10'] = ['9','11']
direct_connection_dict['11'] = ['10','12']
direct_connection_dict['12'] = ['iii','L','S','19']
direct_connection_dict['13'] = ['8','14','16','17'] #17 valt te betwisten
direct_connection_dict['14'] = ['13','15','16'] #16 valt te betwisten
direct_connection_dict['15'] = ['14','16']
direct_connection_dict['16'] = ['13','14','15','17','18','19']
direct_connection_dict['17'] = ['13','16','18','19']
direct_connection_dict['18'] = ['16','17','19']
direct_connection_dict['19'] = ['S','16','17','18'] #moet 'iii' hier ook bij?

def make_connection_dict_and_matrices():
    
    direct_connection_keys = list(direct_connection_dict.keys())
    size = len(direct_connection_keys)
    #print('direct_connection_matrix has dimension ', (size,size))
    
    direct_connection_matrix = np.zeros((size,size))
    
    for i in range(size) :
        current_key = direct_connection_keys[i]
        current_value = direct_connection_dict[current_key]
        
        for j in range(len(current_value)):
            #print(current_value[j], end =" ")
            room_nr = direct_connection_keys.index(current_value[j])
            direct_connection_matrix[i][room_nr] = 1
            direct_connection_matrix[room_nr][i] = 1
        #print("")
    
    np.set_printoptions(threshold=sys.maxsize) # change print size to nothave truncation when printing
    #print(direct_connection_matrix)
    np.set_printoptions(threshold=1000)  # reset print size to default
    
    secondary_connection_matrix = np.zeros((size,size))
    # Voeg M en I als 2e connection manueel toe als ze niet tot de directe connecties behoren
    #index_I = direct_connection_keys.index("I")
    #index_M = direct_connection_keys.index("M")
    #secondary_connection_matrix[index_M][index_I] = 1
    #secondary_connection_matrix[index_I][index_M] = 1
    
    
    for index_current_room in range(size) :
        current_room = direct_connection_keys[index_current_room]
        connected_rooms_to_current = direct_connection_dict[current_room]
        
        for j in range(len(connected_rooms_to_current)):
            #print(current_value[j], end =" ")
            index_secondary_room_in_keys = direct_connection_keys.index(connected_rooms_to_current[j])
            #room_connected_to_current_room = direct_connection_keys[i]
            
            for k in range(len(direct_connection_matrix[index_secondary_room_in_keys])):
                if(direct_connection_matrix[index_secondary_room_in_keys][k] == 1):
                    secondary_connection_matrix[index_current_room][k] = 1
                    secondary_connection_matrix[k][index_current_room] = 1
        #print("")
    
    #keys are necessary to be able to get the room from the index in the matrix
    return direct_connection_keys, direct_connection_matrix, secondary_connection_matrix
""" 
def determine_possible_rooms(painting, N, threshold):
    normalized_scores = best_matching_score.topN_scores(painting) #list of lists of form [name, room, score]

    current_guess = normalized_scores[0]
    highest_score = current_guess[2] #should always be 1 if normalized
    other_possible_rooms = []
    
    for guess in normalized_scores:
        score = guess[2]
        if(score / highest_score > threshold):
            other_possible_rooms.append(guess)
        else:
            break
    
    return current_guess, other_possible_rooms
"""
        
        
    