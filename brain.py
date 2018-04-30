import numpy as np 
import time
from PIL import Image

'''
This is a toy neural network library. 

It has a set 4 layer structure: Input, hidden layer 1, hidden layer 2, and output for a total of 3 synapses. 
This incorporates adding a bias at each layer of the network.

The nodes at each layer are varaible.

Classes:
NN --> Main neural network object
DataCleaner --> Allows you to import an excel file / img

Methods:
NN.train(inpts, outpts, epocs=1000, LR=0.05, error_print_interval=25, activation=None):
NN.predict(inpts)

DataCleaner.load_img(path)
DataCleaner.load_csv(path)
DataCleaner.data = np array
'''

class DataCleaner(object):
    def __init__(self):
        self.data = None

    def load_img(self, img_path, image_matrix=False, normalize=False, in_place=True):
        '''
        image_matrix = True | creates row/col format for images
        normalize = True | divide all pixel values by 255
        in_place = True | does not return the pixel_row (not recommended for batch usage)
        '''

        im = Image.open(img_path)
        im.load()
        pixels = list(im.getdata())
        width, height = im.size

        if image_matrix: # returns the image in row / column format
            pixel_matrix = []
            if not normalize:
                for i in range(height):
                    temp = []
                    for j in range(width):
                        pix = pixels[i * j + j][0]
                        temp.append(pix)
                    pixel_matrix.append(temp)
            else: #Divides all pixels by 255
                for i in range(height):
                    temp = []
                    for j in range(width):
                        pix = pixels[i * j + j][0]
                        pix /= 255
                        temp.append(pix)
                    pixel_matrix.append(temp)
        else: # returns the image as one long string of pixels
            pixel_matrix = []
            if not normalize:
                for i in range(height):
                    for j in range(width):
                        pix = pixels[i * j + j][0]
                        pixel_matrix.append(pix)
            else: #Divides all pixels by 255
                for i in range(height):
                    for j in range(width):
                        pix = pixels[i * j + j][0]
                        pix /= 255
                        pixel_matrix.append(pix)

        if not in_place:
            return self.convert_to_numpy_array(pixel_matrix, flat=True)
        else:
            self.data = self.convert_to_numpy_array(pixel_matrix)

    def convert_to_numpy_array(self, data, flat=False):
        '''Assumes a square array: every column same length'''
        if not flat:
            rows = len(data)
            cols = len(data[0])
            temp = np.zeros((rows, cols))

            for i in range(rows):
                for j in range(cols):
                    temp[i][j] = data[i][j]
        else:
            temp = np.array([data])

        return temp

    def load_csv(self, file_path):
        f = open(file_path)
        
        data = []
        for row in f:
            temp = []
            for elm in row:
                if elm is not '\t' and elm is not '\n' and elm is not ',':
                    temp.append(float(elm))
            data.append(temp)
        f.close()
        
        self.data = self.convert_to_numpy_array(data)


class NN(object):
    def __init__(self, size_tuple=(1, 1, 1, 1)):
        # Matrix: [row, col]
        # The + 1 to the row is to account for the added bias
        self.syn0 = np.random.random((size_tuple[0] + 1, size_tuple[1])) * 2 - 1
        self.syn1 = np.random.random((size_tuple[1] + 1, size_tuple[2])) * 2 - 1
        self.syn2 = np.random.random((size_tuple[2] + 1, size_tuple[3])) * 2 - 1 
        
    def activation(self, z, deriv=False):
        '''The activation function built in is the sigmoid function'''
        sigmoid = 1 / (1 + np.exp(-z))

        if deriv:
            return sigmoid * (1 - sigmoid)
        else:
            return sigmoid

    def addBias(self, array):
        '''Adds a 1 to the end of each row in a given np array'''
        inpt_rows = len(array)
        bias = np.ones((inpt_rows, 1))
        input_with_bias = np.hstack((array, bias))

        return input_with_bias

    def predict(self, inpt):
        ''' Only accepts inputs of numpy arrays'''
        l0 = inpt
        l0 = self.addBias(l0)
        l1 = np.dot(l0, self.syn0)
        l1 = self.activation(l1)

        l1 = self.addBias(l1)
        l2 = np.dot(l1, self.syn1)
        l2 = self.activation(l2)

        l2 = self.addBias(l2)
        l3 = np.dot(l2, self.syn2)
        l3 = self.activation(l3)

        return l3

    def weights_to_csv(self, file_name):
        '''Use this to export the neural networks weight matrix'''
        synapse_list = [self.syn0, self.syn1, self.syn2]
        final_string = ''
        f = open(file_name, 'w')

        for syn in synapse_list:
            final_string += '\n\n'
            for row in syn:
                final_string += '\n'
                for weight in row:
                    final_string += str(weight) + ','
        f.write(final_string)
        f.close()

    def train(self, inpts, outpts, epocs=1000, LR=0.05, error_print_interval=25, activation=None):
        '''
        This method trains the neural network using forward / backward propogation and gradient decent.
        By default, we are using the sigmoid activation function
        '''
        if not activation:
            activation = self.activation

        for i in range(epocs):
            #(3,4,5,1)
            #inpts = (3,3)

            #syn0 = (4, 4)
            #syn1 = (5, 5)
            #syn2 = (6, 1)

            #output = (3,1)

            # 
            # FORWARD PROPOGATION 
            #
            l0 = inpts
            # (3,3) l0
            # (3,3) --> (3,4) L0
            l0 = self.addBias(l0)
            # (3,4) x (4,4) --> (3,4) 
            l1 = np.dot(l0, self.syn0)
            l1 = activation(l1)

            # (3,4) --> (3,5) L1
            l1 = self.addBias(l1)
            # (3,5) x (5,5) --> (3,5) 
            l2 = np.dot(l1, self.syn1)
            l2 = activation(l2)

            # (3,5) -- > (3,6) L2
            l2 = self.addBias(l2)
            # (3,6) x (6,1) --> (3,1) l3
            l3 = np.dot(l2, self.syn2)
            l3 = activation(l3)

            # 
            # BACK PROPOGATION
            # Determine errors

            l3_error = outpts - l3 # (3,1) L3_ERROR
            l3_delta = l3_error * activation(l3, deriv=True)

            # (3,1) x (6,1).T --> (3,6) L2_ERROR
            l2_error = np.dot(l3_error, self.syn2.T)
            l2_delta = l2_error * activation(l2, deriv=True)

            l2_error = l2_error[:,:-1] # to remove bias?
            # (3,5) x (5,5).T --> (3,5) L1_ERROR
            l1_error = np.dot(l2_error, self.syn1.T)
            l1_delta = l1_error * activation(l1, deriv=True)

            # 
            # BACK PROPOGATION
            # Update Weights
            l2_delta = l2_delta[:,:-1] # to remove bias delta
            l1_delta = l1_delta[:,:-1] # to remove bias delta

            #(6,1) += (3,6).T x (3,1) || (6,3) x (3,1) -- > 6,1
            self.syn2 += np.dot(l2.T, l3_delta) * LR
            #(5,5) += (3,5).T x (3,6) || (5,3) x (3,6) --> (5,6)
            self.syn1 += np.dot(l1.T, l2_delta) * LR
            self.syn0 += np.dot(l0.T, l1_delta) * LR

            if i % error_print_interval == 0:
                total_final_layer_error = np.sum(l3_error)
                print("Error: {}".format(str(total_final_layer_error)))
            if i == epocs - 1:
                final_out = l3
                print("Final result: {}".format(str(final_out)))
                time.sleep(3)

if __name__ == '__main__':
    #fake_inpt = np.array([[1,0,1],[1,1,1],[1,0,0]])
    #fake_outpt = np.array([[1,0,1]]).T

    inpt = DataCleaner()
    inpt.load_csv('inpts.csv')
    fake_inpt = inpt.data

    outpt = DataCleaner()
    outpt.load_csv('outpts.csv')
    fake_outpt = outpt.data

    n = NN((3,4,5,1))
    n.train(fake_inpt, fake_outpt, epocs=1000)
