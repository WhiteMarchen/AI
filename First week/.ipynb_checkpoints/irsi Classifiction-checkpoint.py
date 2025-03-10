import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# DT
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# RF
from sklearn.ensemble import RandomForestClassifier

# SVM
from sklearn.svm import SVC

# LF
from sklearn.linear_model import LogisticRegression

file_path = "/Users/marchen/Desktop/programming/AI/First week/iris.csv"
df = pd.read_csv(file_path)

# 타켓 데이터 분리
X = df.iloc[ :, :-1]                            # 특정 데이터
Y = LabelEncoder().fit_transform(df["Name"])    # 문자열을 숫자로 치환

# 데이터 학습, 테스트, 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# DT 모델 학습
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)

# 예측 및 평가
Y_pred = dt_model.predict(X_test)
DT_accuracy = accuracy_score(Y_test, Y_pred)
print(f"DT  예측값 : {Y_pred[:5]}")
print(f"DT  정확도 : {DT_accuracy : .4f}\n")

# RF 모델 학습
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

# 예측 및 평가
Y_pred = rf_model.predict(X_test)
RF_accuracy = accuracy_score(Y_test, Y_pred)
print(f"RF  예측값 : {Y_pred[:5]}")
print(f"RF  정확도 : {RF_accuracy : .4f}\n")

# SVM 모델 학습
svm_model = SVC()
svm_model.fit(X_train, Y_train)

# 예측 및 평가
Y_pred = svm_model.predict(X_test)
SVM_accuracy = accuracy_score(Y_test, Y_pred)
print(f"SVM 예측값 : {Y_pred[:5]}")
print(f"SVM 정확도 : {SVM_accuracy : .4f}\n")

# LF 모델 학습
lr_model = LogisticRegression(max_iter = 200)
lr_model.fit(X_train, Y_train)

# 예측 및 평가
Y_pred = lr_model.predict(X_test)
LF_accuracy = accuracy_score(Y_test, Y_pred)
print(f"LF  예측값 : {Y_pred[:5]}")
print(f"LF  정확도 : {LF_accuracy : .4f}\n")

# 데이터 프레임 확인
print(df.head())
print(df.columns)
print("\n")