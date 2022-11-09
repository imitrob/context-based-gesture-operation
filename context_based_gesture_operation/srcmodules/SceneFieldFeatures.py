
import numpy as np

class SceneFieldFeatures():
    @staticmethod
    def eeff__feature(object_positions, eef_position):
        ''' Deterministically compute eef field based on observation
        '''
        eefobj_distances_ = object_positions - eef_position
        eefobj_distances = np.sum(np.power(eefobj_distances_,2), axis=1)

        eeff = SceneFieldFeatures.gaussian(eefobj_distances)
        return eeff

    @staticmethod
    def feaf__feature(object_sizes, gripper_range):
        ''' Deterministically compute action dependent feature
        Returns: shape = (n_objects)
        '''
        feaf = SceneFieldFeatures.sigmoid(object_sizes, center=gripper_range)
        return feaf

    @staticmethod
    def gaussian(x, sigma=0.2):
        return np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))

    @staticmethod
    def sigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((center-x)*(-tau)))


if __name__ == '__main__':
    object_positions = np.array([[0,0,0], [0,0,0.7]])
    eef_position = np.array([0,0,0])
    eeff = SceneFieldFeatures.eeff__feature(object_positions, eef_position)
    print(eeff)

    object_sizes = np.array([0.1, 0.9, 0.01, 0.15])
    gripper_range = np.array(0.14)
    feaf = SceneFieldFeatures.feaf__feature(object_sizes, gripper_range)
    print(feaf)
