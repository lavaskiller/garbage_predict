import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

################### 이 부분 자동으로 마지막 열 8개를 테스트 데이터로 바꾸는 코드 만들어야 함 ####################
train = pd.read_csv("garbage_train.csv", parse_dates=["date"])
print(train.shape)

test = pd.read_csv("garbage_test.csv", parse_dates=["date"])
print(test.shape)
###########################################################################################################

print(train.info())
#print(train.head(30))

categorical_feature_names = ["plastic", "paper", "general",    # 이 부분 Weather, Rest 추가 예정
 "can", "pet", "glass", "vinyl", "final" , "m_plastic", "m_paper", "m_general", "m_can", "m_pet", "m_glass", "m_vinyl", "m_final"] # "m_plastic", "m_paper", "m_general", "m_can", "m_pet", "m_glass", "m_vinyl", "m_final"

for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

feature_names = ["plastic", "paper", "general",
 "can", "pet", "glass", "vinyl"]

X_train = train[feature_names]
X_test = test[feature_names]

label_name = "m_final"
yx_train = train[label_name]


happy_predict = ["m_plastic", "m_paper", "m_general",
 "m_can", "m_pet", "m_glass", "m_vinyl"]
yy_train = train[happy_predict]


from sklearn.ensemble import RandomForestClassifier

max_depth_list = []

model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

model.fit(X_train, yx_train)
x_predictions = model.predict(X_test)

model.fit(X_train, yy_train)
y_predictions = model.predict(X_test)


def mode(arr):
    counter = Counter(arr)
    if len(counter) == 1: return arr[0]
    counting_arr =counter.most_common(n=2)
    if counting_arr[0][1] == counting_arr[1][1]:
        return counting_arr[1][0]
    return counting_arr[0][0]

final_predict = mode(x_predictions)

if final_predict == 1: print("자동할당: plastic")
if final_predict == 2: print("자동할당: paper")
if final_predict == 3: print("자동할당: general")
if final_predict == 4: print("자동할당: can")
if final_predict == 5: print("자동할당: pet")
if final_predict == 6: print("자동할당: glass")
if final_predict == 7: print("자동할당: vinyl")


print("예상 일별 플라스틱 배출량: {}".format(y_predictions[6,0]))
print("예상 일별 종이 배출량: {}".format(y_predictions[6,1]))
print("예상 일별 일반쓰레기 배출량: {}".format(y_predictions[6,2]))
print("예상 일별 캔 배출량: {}".format(y_predictions[6,3]))
print("예상 일별 페트 배출량: {}".format(y_predictions[6,4]))
print("예상 일별 비닐 배출량: {}".format(y_predictions[6,5]))