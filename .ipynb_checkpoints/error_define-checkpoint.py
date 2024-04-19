import numpy as np
import pandas as pd

def downside_square_error(pred,test):
    N = len(pred)
    total_error = 0
    for i in range(len(pred)):
        total_error += (max(test[i] - pred[i],0))**2
    return (total_error/N)
