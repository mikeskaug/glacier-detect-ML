import rasterio
import numpy as np
import os

CONFIG = {'training_regions': {'LE70322482009163EDC00': {'glacier': [519681, 9011850, 521846, 9013798],  # [minx, miny, maxx, maxy]
                                                         'other': [488853, 9017286, 549727, 9063374]},
                               'LE70322482009195EDC00': {'glacier': [521377, 8986356, 532899, 8998677],
                                                         'other': [519912, 9034509, 531567, 9046897]},
                               'LE70332482010173EDC00': {'glacier': [520936, 8987725, 534057, 8999914],
                                                         'other': [512611, 9075173, 526464, 9089026]},
                               'LE70342482009177EDC00': {'glacier': [529394, 8964215, 539984, 8989657],
                                                         'other': [513760, 9027353, 525615, 9039275]}},
          'label_index': {'glacier': 0, 'other': 1},
          'band_postfix': ['_B1_clipped.TIF', '_B2_clipped.TIF', '_B3_clipped.TIF', '_B4_clipped.TIF',
                           '_B5_clipped.TIF', '_B6_VCID_2_clipped.TIF', '_B7_clipped.TIF'],
          'features': ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7'],
          'train_samples_per_region': 2000,
          'test_samples_per_region': 100,
          'num_images': 4,  # = len(CONFIG['training_regions'])
          'num_categories': 2,
          'image_directory': './images/cropped'
          }


def remove_no_data_pixels(data, labels):
    # remove rows where every element in data == 0 indicating no data
    is_zero = np.equal(data, 0)
    no_data = np.all(is_zero, axis=1)
    culled_data = data[~no_data, ...]
    culled_labels = labels[~no_data, ...]

    return (culled_data, culled_labels)


def construct_features():
    training_set = np.array([]).reshape(0, len(CONFIG['features']))
    training_labels = np.array([]).reshape(0, CONFIG['num_categories'])

    test_set = np.array([]).reshape(0, len(CONFIG['features']))
    test_labels = np.array([]).reshape(0, CONFIG['num_categories'])

    for key, value in CONFIG['training_regions'].items():
        for category, bounds in value.items():
            # select random pixels from each region
            train_x = np.random.randint(bounds[0], bounds[2], CONFIG['train_samples_per_region'])
            train_y = np.random.randint(bounds[1], bounds[3], CONFIG['train_samples_per_region'])
            test_x = np.random.randint(bounds[0], bounds[2], CONFIG['test_samples_per_region'])
            test_y = np.random.randint(bounds[1], bounds[3], CONFIG['test_samples_per_region'])

            train_sub_set = np.zeros([CONFIG['train_samples_per_region'], len(CONFIG['features'])])
            test_sub_set = np.zeros([CONFIG['test_samples_per_region'], len(CONFIG['features'])])

            train_sub_labels = np.zeros([CONFIG['train_samples_per_region'], CONFIG['num_categories']])
            test_sub_labels = np.zeros([CONFIG['test_samples_per_region'], CONFIG['num_categories']])

            train_set_coords = zip(train_x, train_y)
            test_set_coords = zip(test_x, test_y)
            for feature_index, band in enumerate(CONFIG['band_postfix']):
                path = CONFIG['image_directory'] + '/' + key + band
                with rasterio.open(path) as dataset:

                    for index, [x, y] in enumerate(train_set_coords):
                        [[I]] = list(dataset.sample([(x, y)]))
                        train_sub_set[index, feature_index] = I
                        train_sub_labels[index, CONFIG['label_index'][category]] = 1

                    for index, [x, y] in enumerate(test_set_coords):
                        [[I]] = list(dataset.sample([(x, y)]))
                        test_sub_set[index, feature_index] = I
                        test_sub_labels[index, CONFIG['label_index'][category]] = 1

            culled_data, culled_labels = remove_no_data_pixels(train_sub_set, train_sub_labels)
            training_set = np.vstack([training_set, culled_data])
            training_labels = np.vstack([training_labels, culled_labels])

            culled_data, culled_labels = remove_no_data_pixels(test_sub_set, test_sub_labels)
            test_set = np.vstack([test_set, culled_data])
            test_labels = np.vstack([test_labels, culled_labels])

    return (training_set, training_labels, test_set, test_labels)


def save_features(training_set, training_labels, test_set, test_labels, out_dir):
    np.save(os.path.join(os.path.abspath(out_dir), 'training_set'), training_set)
    np.save(os.path.join(os.path.abspath(out_dir), 'training_labels'), training_labels)
    np.save(os.path.join(os.path.abspath(out_dir), 'test_set'), test_set)
    np.save(os.path.join(os.path.abspath(out_dir), 'test_labels'), test_labels)
