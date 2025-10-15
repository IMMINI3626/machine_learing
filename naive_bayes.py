import pandas as pd

# ===== CSV 파일 불러오기 =====
file_name = "naive_data.csv"

try:
    data = pd.read_csv(file_name, sep=",")
except FileNotFoundError:
    print(f"파일 열기 중 오류 발생 : '{file_name}' 파일을 찾을 수 없습니다.\n같은 폴더안에 있는지 확인해주세요.")
    exit()
except pd.errors.EmptyDataError:
    print(f"파일 오류: '{file_name}' 파일이 비어 있습니다.\n파일 내용을 확인해주세요.")
    exit()

# ===== Laplace Smoothing 설정 =====
alpha = 1  # 스무딩 파라미터 값 = 1
target = data['Stolen'].unique()  # 목표 클래스 설정

# ===== 사전확률 계산 =====
class_prob = {c: len(data[data['Stolen'].str.lower() == c.lower()]) / len(data) for c in target}

# ===== 조건부 확률 (라플라스 스무딩) =====
def laplace_smoothing(feature, value, target_class):
    # target_class와 feature 값을 모두 소문자로 비교
    subset = data[data['Stolen'].str.lower() == target_class.lower()]
    n = len(subset)
    n_c = len(subset[subset[feature].str.lower() == value.lower()])
    k = len(data[feature].unique())

    # Laplace Smoothing 공식
    return (n_c + alpha) / (n + alpha * k)

# ===== 예측 함수 =====
def predict(test_data):
    Problem = {}
    for c in target:
        prob = class_prob[c]
        for feature, value in test_data.items():
            prob *= laplace_smoothing(feature, value, c)
        Problem[c] = prob

    print("\n===== probability =====")
    # yes → 1, no → 0
    for c, prob_val in Problem.items():
        label = 1 if c.lower() == "yes" else 0
        print(f"P(y={label}) = {prob_val:.10f}")

    return max(Problem, key=Problem.get)

# ===== 사용자 입력 =====
print("\n===== Example =====")
User_Color = input("Color (Red, Yellow): ").strip().lower()
User_Type = input("Type (Sports, SUV): ").strip().lower()
User_Origin = input("Origin (Domestic, Imported): ").strip().lower()
# .lower()로 대소문자 구분 없이 사용

# ===== 테스트 샘플 생성 =====
test_data = {
    'Color': User_Color,
    'Type': User_Type,
    'Origin': User_Origin
}

# ===== 예측 결과 출력 =====
prediction = predict(test_data)
prediction_label = 1 if prediction.lower() == "yes" else 0

print(f"\n예측 결과: y={prediction_label} ({prediction})")
