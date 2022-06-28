import csv
import numpy as np

class PaintingGroundThruth:
  def __init__(self, room, imagename, painting_number, tl, tr, br, bl):
    self.room = room
    self.imagename = imagename
    self.painting_number = painting_number
    self.tl = tl
    self.tr = tr
    self.br = br
    self.bl = bl

groundtruth_paintings = []

with open('Database_log.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            tl = np.fromstring(row[3].replace("[", "").replace("]", ""), dtype=int, sep=',')
            tr = np.fromstring(row[4].replace("[", "").replace("]", ""), dtype=int, sep=',')
            br = np.fromstring(row[5].replace("[", "").replace("]", ""), dtype=int, sep=',')
            bl = np.fromstring(row[6].replace("[", "").replace("]", ""), dtype=int, sep=',')
            groundtruth_paintings.append(PaintingGroundThruth(row[0], row[1], row[2], tl, tr, br, bl))
            line_count += 1
            
    print(f'Processed {line_count} lines.')

def get_groundtruth_paintings():
    return groundtruth_paintings