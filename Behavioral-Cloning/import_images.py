import csv
import pickle
import cv2
import numpy as np

from sklearn.model_selection import train_test_split


def load_images_to_pickle(valid_ratio, test_ratio, pickle_load_dir, image_limit, steering_correction,
                          log_directory='examples/driving_log.csv'):
    print("Loading images ...")
    with open(log_directory, 'rt') as f:
        reader = csv.reader(f, delimiter=',', )

        images_array = []
        all_images_array = []
        steerings_array = []
        all_steerings_array = []

        i = 0

        for row in reader:
            i += 1
            if 1 < i < image_limit:

                steering_center = float(row[3])
                # steering_left = steering_center - steering_correction
                # steering_right = steering_center + steering_correction

                image_center = cv2.cvtColor(cv2.imread("./examples/" + row[0].strip()), cv2.COLOR_BGR2RGB)
                # image_left = cv2.cvtColor(cv2.imread("./examples/" + row[1].strip()), cv2.COLOR_BGR2RGB)
                # image_right = cv2.cvtColor(cv2.imread("./examples/" + row[2].strip()), cv2.COLOR_BGR2RGB)

                images_array.append(np.array(image_center, dtype='u8'))
                steerings_array.append(np.array(steering_center))

                # all_images_array.append(np.array([image_left, image_center, image_right]))
                # all_steerings_array.append(np.array([steering_left, steering_center, steering_right]))

        images_array = np.array(images_array, dtype='u8')
        # all_images_array = np.array(all_images_array)

    train_valid_test_split(valid_ratio=valid_ratio,
                           test_ratio=test_ratio,
                           x_data=images_array,
                           y_data=steerings_array,
                           load_dir=pickle_load_dir + "center_angle/")

    # train_valid_test_split(valid_ratio=valid_ratio,
    #                        test_ratio=test_ratio,
    #                        x_data=all_images_array,
    #                        y_data=all_steerings_array,
    #                        load_dir=pickle_load_dir + "all_angles/")

    print("Done !")


def train_valid_test_split(valid_ratio, test_ratio, x_data, y_data, load_dir):
    valid_ratio = valid_ratio / (1 - test_ratio)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_ratio, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_ratio, random_state=42)
    with open(load_dir + 'train.pickle', 'wb') as f:
        pickle.dump([x_train, y_train], f)
    with open(load_dir + 'valid.pickle', 'wb') as f:
        pickle.dump([x_valid, y_valid], f)
    with open(load_dir + 'test.pickle', 'wb') as f:
        pickle.dump([x_test, y_test], f)
