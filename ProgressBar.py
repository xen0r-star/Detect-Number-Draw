import sys

def update_progress_bar(iteration, loss, variation, time):
    # length=50
    # progress = str(iteration).zfill(len(str(total)))
    # percent = int(iteration / total * 100)
    # progress_bar = '━' * int(length * iteration / total) + \
    #                ' ' * (length - int(length * iteration / total))
    
    # if percent >= 85:
    #     color = '\033[92m'
    # elif percent >= 50:
    #     color = '\033[93m'
    # else:
    #     color = '\033[91m'


    if loss <= 1:
        color = '\033[92m'
    elif loss <= 10:
        color = '\033[93m'
    else:
        color = '\033[91m'

    sys.stdout.write('\rLearning : \033[96m{}\033[0m | Loss : {}\033[0m | Variation : \033[93m{}\033[0m  | Time : \033[94m{}\033[0m'.format(str(iteration).zfill(13), color + str(round(loss, 14)), str(variation) + "0" * (16 - len(str(variation))), time))
    sys.stdout.flush()