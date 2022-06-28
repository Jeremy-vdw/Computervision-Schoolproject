import db_connect
import best_matching_score
import pickle 

def eval_database_MSE():
    paintings = db_connect.get_paintings()
    eval_score = []
    total_MSE = 0.0
    i = 0
    for painting in paintings.find():
        painting_name = painting["name"]
        painting_room = painting["room"]
        painting_scores = best_matching_score.matching_scores(painting)
        #we calculate the MSE
        MSE = 0.0
        for score in painting_scores:
            if score[0] == painting_name and score[1] == painting_room:
                MSE += pow(1-score[2],2)
            else:
                MSE += pow(score[2], 2)
        
        MSE /= len(painting_scores)
        #add to the eval_scores
        eval_score.append([painting_name, painting_room, MSE])
        print([painting_name, painting_room, MSE])

        total_MSE += MSE
        i += 1
    print("Average MSE: ", total_MSE/i)

def eval_database_ratio():
    paintings = db_connect.get_paintings()
    eval_score = []

    i = 0
    for painting in paintings.find():
        painting_name = painting["name"]
        painting_room = painting["room"]
        painting_scores = best_matching_score.matching_scores(painting)
        #we calculate the ratio
        highest_score = 0.0
        own_score = 0.0

        total_ratio = 1
        i = 0
        for score in painting_scores:
            if score[0] == painting_name and score[1] == painting_room:
                own_score = score[2]
            elif score[2] > highest_score:
                    highest_score = score[2]
        
        ratio = own_score/highest_score
        #add to the eval_scores
        eval_score.append([painting_name, painting_room, ratio])
        print([painting_name, painting_room, ratio])

        total_ratio *= ratio
        i += 1
    print("total ratio: ", pow(total_ratio, 1/i))



def eval_video():
    pass

eval_database_ratio()