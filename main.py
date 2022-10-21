import face_recognition
from pathlib import Path
import os
path=(os.getcwd())
print(path)
known_image = face_recognition.load_image_file("{}\{}".format(path,"IMG_20220905_202725.jpg"))
unknown_image = face_recognition.load_image_file("_DSC0250.JPG")


biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

known_encodings = [
    biden_encoding,
    unknown_encoding
]

# print(len(unknown_encoding))
# print(unknown_encoding)

results = face_recognition.compare_faces([biden_encoding], unknown_encoding[0])
print(results)

image_to_test = face_recognition.load_image_file("IMG_20220905_202713.jpg")
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]

face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

print(face_distances)
print({"first_image" : face_distances[0],"second_image_distance":face_distances[1]})
# for i, face_distance in enumerate(face_distances):
#     print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
#     print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
#     print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
#     print()


#this is the just editing purpose writing query










