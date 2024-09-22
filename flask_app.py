import requests
from flask import Flask, render_template, request, jsonify
import json
import tensorflow as tf
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://43.203.16.219:8080"}})

model_path = "/home/ubuntu/modelServer/src/final_model.h5"
model = tf.keras.models.load_model(model_path) 

# 사용할 액션 레이블 리스트 (모델에서 예측할 클래스)
actions = np.array(['안녕하세요', '사랑합니다', '고맙습니다', '너', '나', '행복합니다', '만나다', '떠나다', '만나서 반갑습니다', '이름'])

# 좌표값을 전처리하는 함수
def preprocess_keypoints(pose_keypoints, left_hand_keypoints, right_hand_keypoints):
    # 키포인트를 flatten하여 배열 생성
    pose = np.array([[kp['x'], kp['y'], kp['z'], kp['visibility']] for kp in pose_keypoints]).flatten() \
        if pose_keypoints else np.zeros(33 * 4)
    lh = np.array([[kp['x'], kp['y'], kp['z']] for kp in left_hand_keypoints]).flatten() \
        if left_hand_keypoints else np.zeros(21 * 3)
    rh = np.array([[kp['x'], kp['y'], kp['z']] for kp in right_hand_keypoints]).flatten() \
        if right_hand_keypoints else np.zeros(21 * 3)
    
    # 모든 키포인트 결합
    keypoints = np.concatenate([pose, lh, rh])

    # shape를 (30, 258)로 맞추기 위해 패딩 또는 트리밍
    if keypoints.shape[0] < 30 * 258:
        # 패딩 추가
        padding = np.zeros((30 * 258 - keypoints.shape[0],))
        keypoints = np.concatenate([keypoints, padding])
    else:
        # 잘라내기
        keypoints = keypoints[:30 * 258]

    return keypoints.reshape(1, 30, 258)  # shape: (1, 30, 258)


# 예측 수행 함수
def infer_action(model, pose_keypoints, left_hand_keypoints, right_hand_keypoints):
    keypoints = preprocess_keypoints(pose_keypoints, left_hand_keypoints, right_hand_keypoints)
    
    # 예측 수행
    predictions = model.predict(keypoints)
    
    # 가장 높은 확률을 가진 액션 선택
    predicted_action = actions[np.argmax(predictions)]
    
    return predicted_action

def load_json_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    json_url = data['s3url'].strip('\"')
    
    try:
        json_data = load_json_from_url(json_url)
    except Exception as e:
        return jsonify({'error': str(e)})

    # 예측 수행
    predicted_meaning = infer_action(
        model,
        json_data["pose_keypoint"][0], 
        json_data["left_hand_keypoint"][0], 
        json_data["right_hand_keypoint"][0]
    )    
    # 응답 반환
    response = jsonify(predicted_meaning)
    response.headers.add('Content-Type', 'application/json; charset=utf-8')

    return app.response_class(
        response=json.dumps(predicted_meaning, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)