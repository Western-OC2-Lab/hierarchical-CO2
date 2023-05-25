import numpy as np
import math

class GAF_New:


    def __init__(self) -> None:
        pass

    def __call__(self, scaled_series):

        scaled_series = np.where(scaled_series >= 1.0, 1.0, scaled_series)
        scaled_series = np.where(scaled_series <= -1.0, -1.0, scaled_series)

        phi = np.arccos(scaled_series)
        gaf = self.tabulate(phi, phi, self.cos_sum)

        return gaf

    
    def tabulate(self, x, y, f):
        return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

    def cos_sum(self, a, b):
        return (math.cos(a+b))

    


        