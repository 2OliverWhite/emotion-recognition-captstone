import re
import numpy as np
import os


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# AF:0 HB:1 SP:2
# neg:0 neu:1 pos:2
# ex_pos:9 ve_pos:8 pos:7 sli_pos:6 neu:5 sli_neg:4 neg:3 ve_neg:2 ex_neg:1
def get_feature(PATH):
    training_features = None
    training_labels = []

    maxCount = 100

    # Root Path Ex ./features/train3
    for class_dir in os.listdir(PATH):
        class_path = os.path.join(PATH, class_dir)
        class_label = int(class_dir) - 1

        count = 0
        for feature_f_name in os.listdir(class_path):
            if count > maxCount:
                break
            count += 1
            feature_path = os.path.join(class_path, feature_f_name)

            features = np.load(feature_path)
            labels = None
            if training_features is None:
                training_features = features
                labels = [class_label for i in range(len(training_features))]
            else:
                labels = [class_label for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)

            training_labels.extend(labels)

       
    print(training_features.shape)
    return training_features, np.array(training_labels).squeeze()
