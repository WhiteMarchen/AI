import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "/Users/marchen/Desktop/programming/AI/First week/iris.csv"
df = pd.read_csv(file_path)

# 타겟 데이터 분리
X = df.iloc[ :, :-1]                            # 특정 데이터
Y = LabelEncoder().fit_transform(df["Name"])    # 문자열을 숫자로 치환

# 데이터 학습, 테스트, 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 선언
models = {
    "DT": DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    "SVM": SVC(),
    "LR": LogisticRegression(max_iter=200)
}

# 모델 학습, 평가
for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"{name}  예측값 : {Y_pred[:5]}")
    print(f"{name}  정확도 : {accuracy:.4f}\n")

print(df.head())
print(df.columns)