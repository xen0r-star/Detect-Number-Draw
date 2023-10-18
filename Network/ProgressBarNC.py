import sys

def update_progress_bar(iteration, loss, variation, time):
    sys.stdout.write('\rLearning : {} | Loss : {} | Variation : {:.15f}  | Time : {}'.format(str(iteration).zfill(13), str(round(loss, 14)), variation, time))
    sys.stdout.flush()