from brain import NN, DataCleaner
import numpy as np
import time
'''
Methods:
NN.train(inpts, outpts, epocs=1000, LR=0.05, error_print_interval=25, activation=None):
NN.predict(inpts)
NN.weights_to_csv(path_name)

load_img(self, img_path, image_matrix=False, normalize=True, in_place=False):
DataCleaner.load_csv(path)
DataCleaner.data = np array

Images are 125x125
'''

network = NN((125*125, 290, 290, 10))
dataCleaner = DataCleaner()

# create testing data
test_data = dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/nine_1.png',
    image_matrix=False, normalize=True, in_place=False)

extra_two = dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/two_2.png',
    image_matrix=False, normalize=True, in_place=False)

test_data = np.vstack((test_data, extra_two))

# load images from the number_pictures directory folder
for j in range(5):
    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/zero_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/one_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/two_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/three_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/four_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))
    
    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/five_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/six_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/seven_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/eight_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

    path = 'C:/Users/Nick/Desktop/image_recognition/number_pictures/nine_{}.png'.format(str(j+1))
    test_data = np.vstack((test_data, dataCleaner.load_img(path,image_matrix=False, normalize=True, in_place=False)))

#[two, six, nine]
# 9 2 0,1,2,3,4,5,6,7,8,9
def create_data_labels(digits, frequency):
    insert_position = 0

    data_labels = np.zeros((digits * frequency, digits))
    for i in range(digits):
        for j in range(frequency):
            data_labels[i * frequency + j][insert_position] = 1
        insert_position += 1
    return data_labels

data_labels = create_data_labels(10, 5) 

additional_labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
data_labels = np.vstack((additional_labels, data_labels))

'''
# Initial Guess
guess_two =  dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/guess_two.png',
    image_matrix=False, normalize=True, in_place=False)
guess_nine = dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/guess_nine.png',
    image_matrix=False, normalize=True, in_place=False)
guess_two2 =  dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/guess_two2.png',
    image_matrix=False, normalize=True, in_place=False)
guess_nine2 = dataCleaner.load_img('C:/Users/Nick/Desktop/image_recognition/number_pictures/guess_nine2.png',
    image_matrix=False, normalize=True, in_place=False)

guess1 = network.predict(guess_two)
guess2 = network.predict(guess_nine)
guess3 = network.predict(guess_two2)
guess4 = network.predict(guess_nine2)

print("Should be a [1,0,0]: ", guess1)
print("\nShould be a [0,0,1]: ", guess2)
print("Should be a [1,0,0]: ", guess3)
print("\nShould be a [0,0,1]: ", guess4)
'''

# Train the Network
network.train(test_data, data_labels, epocs=16000, LR=0.0003, error_print_interval=2000)

'''
# Educated Guess
guess1 = network.predict(guess_two)
guess2 = network.predict(guess_nine)
guess3 = network.predict(guess_two2)
guess4 = network.predict(guess_nine2)

print("Should be a [1,0,0]: ", guess1)
print("\nShould be a [0,0,1]: ", guess2)
print("Should be a [1,0,0]: ", guess3)
print("\nShould be a [0,0,1]: ", guess4)
'''