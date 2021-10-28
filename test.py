from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd
'''
img1 = DeepFace.detectFace("jhlee_1.jpg")
img2 = DeepFace.detectFace("jhlee_2.jpg")
plt.imshow(img1)
plt.savefig("test.jpg")
plt.imshow(img2)
plt.savefig("test1.jpg")
'''

obj = DeepFace.analyze("jhlee_1.jpg")
print(obj)


'''
result = DeepFace.verify(img1_path = "jhlee_1.jpg", img2_path = "jhlee_2.jpg", model_name = "Facenet")
print(result)
'''
