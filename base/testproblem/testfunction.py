import math

class Rastrigin():

    def __init__(self, dim):
        self.dim = dim
    def __call__(self, input_x):
        f_x = 10. * len(input_x)
        for i in input_x.values():
            f_x += i**2 - 10 * math.cos(2*math.pi*i)
        return f_x


    def _load_api_config(self):
        return {
            'x{}'.format(k): [-5.12, 5.12] for k in range(self.dim)
        }