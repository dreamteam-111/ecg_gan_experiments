import numpy as np


def get_interpolation(num_steps, p1, p2):

    result = []
    latent_dim = len(p1)
    for i in range(latent_dim):
        c_1_varied = np.linspace(p1[i], p2[i], num_steps)
        result.append(c_1_varied)
    return np.array(result)



def get_ci_varied_over_zeros(num_steps=5, latent_dim=11, i=0):
    zeros = np.zeros((num_steps, latent_dim-1))
    values = np.linspace(-1, 1, num_steps)
    b = np.insert(zeros, i, values, axis=1)
    return b

if __name__ == "__main__":
    p1 = np.array([0,1,2])
    p2 = np.array([5,1,10])
    #print (get_interpolation(5,p1,p2))


