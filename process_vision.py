import os
import pickle
import random
import numpy as np

from tqdm import tqdm
from id_split import train_list, val_list, test_list


def KL(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

demographics = open("data/demographics_0216.csv", "r").readlines()[1:]
demographics_dict = {}
for row in demographics:
    id, name, gender, ethnicity = row[:-1].split(", ")
    demographics_dict[id] = [gender, ethnicity]

with open("data/voxceleb2_wavlm_inference.pkl", "rb") as f:
    expression_labels = pickle.load(f)

annotation_file = open(f"data/text_emotion_0116.csv", "r").readlines()[1:]

# print(len(train_list), len(val_list), len(test_list)) # 3208 442 931

train_data, val_data, test_data = [], [], []
emotion_list = []
all_data = {}

for annotation in tqdm(annotation_file):
    annotation = annotation[:-1].split(", ")
    transcript_path = annotation[0]
    subject_id = transcript_path.split("/")[2]
    transcript_path = transcript_path.replace("./transcript/", "").replace("txt", "wav")

    # anger, disgust, fear, joy, neutral, sadness, surprise
    # neutral, anger, disgust, fear, happiness, sadness, surprise
    text_emotion = [float(annotation[5]), float(annotation[1]), float(annotation[2]), float(annotation[3]), float(annotation[4]), float(annotation[6]), float(annotation[7])]
    # text_emotion = [float(annotation[1]), float(annotation[2]), float(annotation[3]), float(annotation[4]), float(annotation[5]), float(annotation[6]), float(annotation[7])]

    key = "dev/mp4/" + transcript_path[:-4]
    if not key in expression_labels:
        continue
    # neutral, anger, disgust, fear, happiness, sadness, surprise
    expression_emotion = expression_labels[key][0]
    expression_emotion = [expression_emotion[7], expression_emotion[0], expression_emotion[5], expression_emotion[4], expression_emotion[2], expression_emotion[1], expression_emotion[3]]
    expression_emotion = np.array(expression_emotion)
    expression_emotion = expression_emotion / np.sum(expression_emotion)

    text_emotion = np.array(text_emotion)
    expression_emotion = np.array(expression_emotion)

    kld = KL(text_emotion, expression_emotion)
    if kld > 1:
        continue
    emotion = (text_emotion + expression_emotion) / 2
    emotion = np.argmax(emotion)

    if emotion == 0:
        emotion = 0
    elif emotion == 1:
        emotion = 1
    elif emotion == 4:
        emotion = 2
    elif emotion == 6:
        emotion = 3
    else:
        continue

    # if not os.path.exists(f"../../data/vox2/audio/{transcript_path}"):
    #     continue
    if not os.path.exists(f"/data/perception-temp/yufeng/backup_20240515/yin/per_libreface/data/vox2/audio/{transcript_path}"):
        continue
    if subject_id not in all_data:
        all_data[subject_id] = []
    all_data[subject_id].append([transcript_path, subject_id, emotion])

new_speaker_dict = {}
utterances_per_speaker = 50
for subject_id in all_data:
    if "n0" + subject_id[2:] not in demographics_dict:
        continue
    data_list = all_data[subject_id]
    keep_list = [1] * len(data_list)
    emotion_list = [x[2] for x in data_list]
    emotion_list = np.array(emotion_list)

    for i, emotion in enumerate(emotion_list):
        p = random.uniform(0, 1)
        if emotion == 1 or emotion == 3:
            continue
        if sum(keep_list) <= utterances_per_speaker:
            continue
        if emotion == 0:
            if p > 10000 / 150123:
                keep_list[i] = 0
        if emotion == 2:
            if p > 10000 / 61386:
                keep_list[i] = 0

    if sum(keep_list) < utterances_per_speaker:
        continue
    new_speaker_dict[subject_id] = sum(keep_list)
    for i, data in enumerate(data_list):
        transcript_path, subject_id, emotion = data
        if keep_list[i] == 0:
            continue
        if subject_id in train_list:
            train_data.append([transcript_path, subject_id, emotion])
        elif subject_id in val_list:
            val_data.append([transcript_path, subject_id, emotion])
        else:
            test_data.append([transcript_path, subject_id, emotion])

print(len(train_data) + len(val_data) + len(test_data))

print(len(new_speaker_dict))
print(min(new_speaker_dict.values()))
print(max(new_speaker_dict.values()))
print(sum(new_speaker_dict.values())/len(new_speaker_dict))

# 0208 KLD 1
# neutral, anger, happiness, surprise
# 53171, 2745, 21550, 2634
# 80100 utterances, 801 speakers

# 0213 KLD 0.7
# neutral, anger, happiness, surprise
# 38046, 2039, 19476, 2309
# 61870 utterances, 1546 speakers
# min 40, max 58, avg 40.02

# 0216 KLD 1
# neutral, anger, happiness, surprise
# 45288, 3682, 21466, 3664
# 74100 utterances, 1480 speakers
# min 50, max 88, avg 50.07

# 0330 KLD 1 vision
# neutral, anger, happiness, surprise
# 39774, 6909, 19168, 9259
# 75110 utterances, 1494 speakers
# min 50, max 105, avg 50.27
