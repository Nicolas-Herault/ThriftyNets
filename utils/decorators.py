################################################  TIMER  ################################################

from time import time

def timed(function):
    def wrapper(*args, **kwargs):
        start = time()
        output = function(*args, **kwargs)
        end = time()
        run_time = end-start
        print('Function mesured...: ' + str(function))
        print('Time taken.........: %dh %dm %.2fs' % (run_time//3600, (run_time%3600)//60, run_time%60))
        return output
    return wrapper

        
