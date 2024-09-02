from flask import Flask, render_template, request, jsonify
import torch
import json
import pickle
from model_class import SignLanguageTranslationModel
import requests

app = Flask(__name__)

model_path = "/home/ubuntu/modelServer/src/trainAll_state.pth"
#model = torch.load(model_path, map_location=torch.device('cpu'))
# 모델 객체 생성
pose_input_dim = 4  # 적절한 입력 차원 설정
hand_input_dim = 6
meaning_input_dim = 5000
hidden_dim = 512
output_dim = 5000
model = SignLanguageTranslationModel(pose_input_dim, hand_input_dim, meaning_input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  

meaning_dict_path = "/home/ubuntu/modelServer/src/meaning_dict.pkl"
with open(meaning_dict_path, 'rb') as f:
  meaning_dict = pickle.load(f)


def preprocess_keypoints(keypoints, input_dim, target_num_keypoints):
    if not keypoints:  # 빈 배열일 경우
        return torch.zeros((1, target_num_keypoints, input_dim))  # 0으로 채운 텐서 반환

    x = [kp['x'] for kp in keypoints]
    y = [kp['y'] for kp in keypoints]
    z = [kp['z'] for kp in keypoints]

    if input_dim == 4:
        visibility = [kp.get('visibility', 1.0) for kp in keypoints]
        tensor = torch.tensor([x, y, z, visibility]).float().unsqueeze(0)  # (1, 4, num_keypoints)
    else:
        tensor = torch.tensor([x, y, z]).float().unsqueeze(0)  # (1, 3, num_keypoints)
    # 패딩 적용
    num_keypoints = tensor.size(2)
    if num_keypoints < target_num_keypoints:
        padding = torch.zeros((1, tensor.size(1), target_num_keypoints - num_keypoints)).float()
        tensor = torch.cat((tensor, padding), dim=2)
    elif num_keypoints > target_num_keypoints:
        tensor = tensor[:, :, :target_num_keypoints]  # 넘치는 부분 잘라냄

    return tensor.transpose(1, 2)  # (1, target_num_keypoints, input_dim)

def infer_meaning(model, pose_keypoints, left_hand_keypoints, right_hand_keypoints, meaning_dict):
    max_num_keypoints_pose = 33  # 포즈의 최대 키포인트 수
    max_num_keypoints_hand = 21  # 손의 최대 키포인트 수

    pose_tensor = preprocess_keypoints(pose_keypoints, input_dim=4, target_num_keypoints=max_num_keypoints_pose)
    left_hand_tensor = preprocess_keypoints(left_hand_keypoints, input_dim=3, target_num_keypoints=max_num_keypoints_hand)
    right_hand_tensor = preprocess_keypoints(right_hand_keypoints, input_dim=3, target_num_keypoints=max_num_keypoints_hand)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 입력 텐서를 각각의 입력으로 모델에 전달
    pose_tensor = pose_tensor.to(device)
    left_hand_tensor = left_hand_tensor.to(device)
    right_hand_tensor = right_hand_tensor.to(device)

    # 의미 입력이 비어있다면, 기본 텐서 생성
    meaning_inputs = torch.zeros((1, 1, 5000)).to(device)

    with torch.no_grad():
        outputs = model(pose_tensor, torch.cat((left_hand_tensor, right_hand_tensor), dim=2), meaning_inputs)

    # 소프트맥스를 통해 확률 분포로 변환
    probabilities = torch.softmax(outputs, dim=1)

    # 가장 높은 확률을 가진 클래스 예측
    predicted_class_index = torch.argmax(probabilities, dim=1).item()

    # 예측된 의미 출력
    predicted_meaning = None
    for meaning, index in meaning_dict.items():
      if index == predicted_class_index:
        predicted_meaning = meaning
        break

    return predicted_meaning

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
    
    # URL에서 JSON 데이터를 가져오는 부분
    #json_url = r"https://hand-coordinates-json.s3.ap-northeast-2.amazonaws.com/%EB%8B%B5_%EA%B3%A0%EB%AF%BC.json"

    try:
        json_data = load_json_from_url(json_url)
    except Exception as e:
        return jsonify({'error': str(e)})

    # 예측 수행
    predicted_meaning = infer_meaning(model, json_data["pose_keypoint"][0], json_data["left_hand_keypoint"][0], json_data["right_hand_keypoint"][0], meaning_dict)
    
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

