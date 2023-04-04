from ultralytics import YOLO
from PIL import Image
import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
import os
import sys

import json

from scripts import pose_estimator
from scripts.classifier import classify_by_hip_distance


# Load the model
model = YOLO(r"E:/GitHub/porsche-students-contest/best.pt")

# Load utils
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Read Image
#source = Image.open(r"E:\GitHub\porsche-students-contest\images\input4.jpeg")

frame_index =-1
cap = cv2.VideoCapture("E:\GitHub\porsche-students-contest\images\IMG-6520.mp4")

if not cap.isOpened():
    print("Error opening video stream or file")

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame_index += 1
        if frame_index % 10 == 0:
            frame_copy = frame.copy()
            results = model.predict(source=frame, save=True)

            crosswalks = []
            persons = []
            predicts_list = []
            predicts_dict = {}

            for idx, box in enumerate(results[0].boxes.boxes):
                if box[-1] == 1:
                    x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    crosswalks.append((x_min, y_min, x_max, y_max))
                elif box[-1] == 2:
                    x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    persons.append((x_min, y_min, x_max, y_max))

            output_photo = Image.open(
                os.path.join(r'E:/GitHub/porsche-students-contest/runs/detect/predict44/' + results[0].path))
            output_photo = np.asarray(output_photo)

            if len(crosswalks) == 0:
                predicts_list.append("nici un pericol\n")
            elif len(crosswalks) == 1:
                predicts_list.append("crosswalk 1\n")
                if len(persons) == 0:
                    predicts_list.append("nici un pericol\n")
                    predicts_dict["crosswalk1"] = "nici un pericol"
                for idx, person in enumerate(persons):
                    if crosswalks[0][0] < person[0] < crosswalks[0][2] and crosswalks[0][1] < person[3] < crosswalks[0][3]:
                        predicts_dict[f"crosswalk1_person{idx}"] = "danger, pe trecere"
                        predicts_list.append("danger, pe trecere\n")
                        output_photo = cv2.putText(output_photo,
                                                   "danger, pe trecere",
                                                   (person[0], person[3] + 5),
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   1,
                                                   (255,0,0),
                                                   2,
                                                   cv2.LINE_AA)
                    elif crosswalks[0][1] < person[3] + 15 < crosswalks[0][3] or \
                            crosswalks[0][1] < person[3] - 15 < crosswalks[0][3]:
                        if person[0] < crosswalks[0][2] + 300:
                            # Get orientation :
                            landmarks = pose_estimator.get_pose_landmarks(np.asarray(frame_copy)[persons[0][1]:persons[0][3], persons[0][0]:persons[0][2]])
                            if landmarks is not None:
                                hips_label = classify_by_hip_distance(landmarks)
                                if hips_label == "side":
                                    predicts_list.append("warning, pe langa trecere\n")
                                    predicts_dict[f"crosswalk1_person{idx}"] = "warning, pe langa trecere"
                                    output_photo = cv2.putText(output_photo,
                                                               "warning, pe langa trecere",
                                                               (person[0], person[3] + 5),
                                                               cv2.FONT_HERSHEY_SIMPLEX,
                                                               1,
                                                               (255, 255, 0),
                                                               2,
                                                               cv2.LINE_AA)
                                else:
                                    predicts_list.append("langa trecere, dar nu se uita in stanga sau dreapta\n")
                                    predicts_dict[f"crosswalk1_person{idx}"] = "langa trecere, dar nu se uita in stanga sau dreapta"
                                    output_photo = cv2.putText(output_photo,
                                                               "no problem, pe langa trecere",
                                                               (person[0], person[3] + 5),
                                                               cv2.FONT_HERSHEY_SIMPLEX,
                                                               1,
                                                               (0, 255, 0),
                                                               2,
                                                               cv2.LINE_AA)
                    else:
                        predicts_list.append("nici un pericol\n")
                        predicts_dict[f"crosswalk1_person{idx}"] = "langa trecere, dar nu se uita in stanga sau dreapta"
                        output_photo = cv2.putText(output_photo,
                                                   "no problem, pe langa trecere",
                                                   (person[0], person[3] + 5),
                                                   cv2.FONT_HERSHEY_SIMPLEX,
                                                   1,
                                                   (0, 255, 0),
                                                   2,
                                                   cv2.LINE_AA)
            elif len(crosswalks) > 1:
                for idx_crs, crosswalk in enumerate(crosswalks):
                    predicts_list.append(f'crosswalk {idx_crs}:\n')
                    if len(persons) == 0:
                        predicts_list.append("nici un pericol\n")
                        predicts_dict[f"crosswalk{idx_crs}"] = "nici un pericol"
                    for idx_prs, person in enumerate(persons):
                        if crosswalk[0] < person[0] < crosswalk[2] and crosswalk[1] < person[3] < crosswalk[3]:
                            predicts_list.append("danger, pe trecere\n")
                            predicts_dict[f"crosswalk{idx_crs}_person{idx_prs}"] = "danger, pe trecere"
                        elif crosswalk[1] < person[3] + 15 or crosswalk[3] > person[3] - 15:
                            if person[0] < crosswalk[2] + 350:
                                # Get orientation :
                                landmarks = pose_estimator.get_pose_landmarks(np.asarray(frame_copy)[persons[0][1]:persons[0][3], persons[0][0]:persons[0][2]])
                                if landmarks is not None:
                                    hips_label = classify_by_hip_distance(landmarks)
                                    if hips_label == "side":
                                        predicts_list.append("warning, pe langa trecere\n")
                                        predicts_dict[f"crosswalk{idx_crs}_person{idx_prs}"] = "warning, pe langa trecere"
                                    else:
                                        predicts_list.append("langa trecere, dar nu se uita in stanga sau dreapta\n")
                                        predicts_dict[
                                            f"crosswalk{idx_crs}_person{idx_prs}"] = "langa trecere, dar nu se uita in stanga sau dreapta"
                        else:
                            predicts_list.append("nici un pericol")
                            predicts_dict[
                                f"crosswalk{idx_crs}_person{idx_prs}"] = "nici un pericol"

            with open(f'video_output/frame{frame_index}_image_out.json', 'w') as outfile:
                json.dump(predicts_dict, outfile)

            file = open(f'video_output/frame{frame_index}_image_out4.txt', 'w')
            file.writelines(predicts_list)
            file.close()

            Image.fromarray(output_photo).save(f'E:/GitHub/porsche-students-contest/runs/detect/predict44/' + results[0].path)

            output_photo = Image.open(os.path.join(r'E:/GitHub/porsche-students-contest/runs/detect/predict44/' + results[0].path))
            output_photo = output_photo.resize((720, 420))
            output_photo = np.asarray(output_photo)
            output_photo = cv2.cvtColor(output_photo, cv2.COLOR_BGR2RGB)
            cv2.imshow('Frame', output_photo)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
