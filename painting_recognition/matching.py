import db_import as db
import feature_extraction as fe
import feature_compare as fc
import time

#score zoals vroeger
''
def calculate_score(keypoints, max_keypoints, histogram_score, min_histogram_score, max_histogram_score):
	keypoints = keypoints / max_keypoints
	histogram_score = 1 - ((histogram_score - min_histogram_score) / (max_histogram_score - min_histogram_score))
	return (keypoints + histogram_score) / 2

#score van enkel keypoints
def calculate_score_keypoints(keypoints, max_keypoints):
	keypoints = keypoints / max_keypoints
	return keypoints

#compare met enkel keypoints miss terug histogrammen toevoegen
def compare(painting):


	pkeys, pdesc = fe.get_SIFTkeypoints(painting)
	histograms = fe.get_part_histograms(painting)
	
	#list with feature scores of each image of db compared to the painting
	list_name_feature_scores = []
	max_keypoints = 0
	min_histogram_score = 0
	max_histogram_score = 0

	#compare with all other images from db
	for image in db.get_paintings():

		histogram_score = image.compare_with_histograms(histograms)
		keypoints = fc.compare_descriptors(image.descriptors, pdesc)

		#list_name_feature_scores.append((image.name, image.room, rgb_histogram_score, lbp_histogram_score, keypoints))
		list_name_feature_scores.append((image.name, image.room, keypoints, histogram_score))

		if keypoints > max_keypoints:
			max_keypoints = keypoints
		
		if histogram_score < min_histogram_score:
			min_histogram_score = histogram_score

		if histogram_score > max_histogram_score:
			max_histogram_score = histogram_score


	if max_keypoints == 0:
		return []


	#list with total scores of each image 
	list_name_totalscore = []

	for name_score in list_name_feature_scores:
		calculated_score = calculate_score(name_score[2], max_keypoints, name_score[3], min_histogram_score, max_histogram_score)
		#?                           naam            zaal            totale score      
		list_name_totalscore.append((name_score[0], name_score[1], calculated_score))

	#sorts of score?
	list_name_totalscore = sorted(list_name_totalscore, key = lambda x: x[2], reverse = True)#[:10]
	return list_name_totalscore


#compare met enkel keypoints miss terug histogrammen toevoegen
def compare_histograms(painting):

	#print("starting to compare a painting.")
	t1 = time.time()

	#extracting features from painting
	histograms = fe.get_part_histograms(painting)

	#list with feature scores of each image of db compared to the painting
	results = []

	#time1 = time.time()
	#compare with all other images from db
	for image in db.get_paintings():
		score = image.compare_with_histograms(histograms)
		results.append((image.name, image.room, score))
		
	#print("finished to db")
	#print("time: %s seconds" % (time.time()-time1))

	results = sorted(results, key = lambda x: x[2], reverse = False)

	t2 = time.time()
	diff = t2 - t1
	print("total time: %s seconds" % (diff))

	return results