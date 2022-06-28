import connectiematrix
import numpy as np
number_of_paintings = {
	"Zaal_1": 17, #10
	"Zaal_2": 36, #18
	"Zaal_3": 0,
	"Zaal_4": 0,
	"Zaal_5": 8, #9
	"Zaal_6": 11, #10
	"Zaal_7": 12, #11
	"Zaal_8": 10,
	"Zaal_9": 6,
	"Zaal_10": 15,
	"Zaal_11": 10,
	"Zaal_12": 4,
	"Zaal_13": 13,
	"Zaal_14": 14,
	"Zaal_15": 58,
	"Zaal_16": 5,
	"Zaal_17": 9,
	"Zaal_18": 16,
	"Zaal_19": 220, #71
	"Zaal_V": 19,
	"Zaal_S": 114, #65
	"Zaal_R": 8, 
	"Zaal_P": 14,
	"Zaal_Q": 1,
	"Zaal_O": 3, #6
	"Zaal_N": 6,
	"Zaal_M": 19,
	"Zaal_L": 11,
	"Zaal_K": 11, #8
	"Zaal_J": 10,
	"Zaal_I": 7, #0
	"Zaal_iii": 0,
	"Zaal_II": 2, #4
	"Zaal_i": 0, #4
	"Zaal_H": 4, #5
	"Zaal_G": 11,
	"Zaal_F": 11, #12
	"Zaal_E": 18,
	"Zaal_D": 18,
	"Zaal_C": 24, #3
	"Zaal_B": 9,
	"Zaal_A": 13
}

#rooms = ["Zaal_1","Zaal_2","Zaal_3","Zaal_4","Zaal_5","Zaal_6","Zaal_7","Zaal_8","Zaal_9","Zaal_10","Zaal_11","Zaal_12","Zaal_13","Zaal_14","Zaal_15","Zaal_16","Zaal_17","Zaal_18","Zaal_19","Zaal_V","Zaal_S","Zaal_R","Zaal_P","Zaal_Q","Zaal_O","Zaal_N","Zaal_M","Zaal_L","Zaal_K","Zaal_J","Zaal_II","Zaal_I","Zaal_H","Zaal_G","Zaal_F","Zaal_E","Zaal_D","Zaal_C","Zaal_B","Zaal_A"]
rooms = ["Zaal_i","Zaal_II","Zaal_iii","Zaal_A","Zaal_B","Zaal_C","Zaal_D","Zaal_E","Zaal_F","Zaal_G","Zaal_H","Zaal_I","Zaal_J","Zaal_K","Zaal_L","Zaal_M","Zaal_N","Zaal_O","Zaal_P","Zaal_Q","Zaal_R","Zaal_S","Zaal_1","Zaal_2","Zaal_3","Zaal_4","Zaal_5","Zaal_6","Zaal_7","Zaal_8","Zaal_9","Zaal_10","Zaal_11","Zaal_12","Zaal_13","Zaal_14","Zaal_15","Zaal_16","Zaal_17","Zaal_18","Zaal_19"]

size = len(rooms)
total_paintings = sum(number_of_paintings.values())

#init transitiematrix
keys, connectiematrix, sec = connectiematrix.make_connection_dict_and_matrices()
transitiematrix = np.zeros((size,size))

for i in range(size):
	total = 0.0
	#probabilities of being in other rooms the next timestep
	for j in range(size):            
		if (connectiematrix[i][j] == 1):# or (i == j):
			paintings = 1 #number_of_paintings[rooms[j]]
			total += paintings
			transitiematrix[i][j] = paintings
		elif (i == j):
			paintings = 100 #number_of_paintings[rooms[j]]
			total += paintings
			transitiematrix[i][j] = paintings

	#normalizing
	for j in range(size):
		transitiematrix[i][j] /= total#make sure this is a double

def room_probabilities(prev_probs):
	#temp = sorted(prev_probs, key=prev_probs.get, reverse=True)[:5]
	#print("1: %s : %0.3f --- 2: %s : %0.3f --- 3: %s : %0.3f --- 4: %s : %0.3f --- 5: %s : %0.3f" % (temp[0],prev_probs[temp[0]],temp[1],prev_probs[temp[1]],temp[2],prev_probs[temp[2]],temp[3],prev_probs[temp[3]],temp[4],prev_probs[temp[4]]))# , end='\r')
	
	new_probs = prev_probs
	for i in range(size):
		prob = 0.0
		for j in range(size):
			prob += (prev_probs[rooms[j]] * transitiematrix[j][i]) #formula on slide HMM ufora      
		new_probs[rooms[i]] = prob
	
	#temp = sorted(new_probs, key=new_probs.get, reverse=True)[:5]
	#print("1: %s : %0.3f --- 2: %s : %0.3f --- 3: %s : %0.3f --- 4: %s : %0.3f --- 5: %s : %0.3f" % (temp[0],new_probs[temp[0]],temp[1],new_probs[temp[1]],temp[2],new_probs[temp[2]],temp[3],new_probs[temp[3]],temp[4],new_probs[temp[4]]))# , end='\r')

	return new_probs

emission_probs = np.load("emission_prob_2.npy", allow_pickle=True).item()

def room_painting_probs(room_probs, painting, painting_room):
	for room in rooms:
		room_probs[room] *= laplace_smoothing(room, painting, painting_room)
	return room_probs

def room_given_painting_probs(prev_probs, painting, painting_room):
	room_probs = room_probabilities(prev_probs)
	room_painting_prob = room_painting_probs(room_probs, painting, painting_room)
	total = 0
	for room in rooms:
		total += room_painting_prob[room]
	
	for room in rooms:
		room_painting_prob[room] = room_painting_prob[room]/total
	return room_painting_prob

def laplace_smoothing(room, painting, painting_room, smoothing = 0.0001):
	try:
		total_paintings_seen = sum(emission_probs[room].values())
		value  = 0

		name = "Zaal_" + painting_room + "_" + painting
		#if painting has been observed
		if name in emission_probs[room]:
			value = emission_probs[room][name]
		
		return (value + smoothing)/(total_paintings_seen + smoothing * total_paintings)
	except:
		return smoothing