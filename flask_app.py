from flask import Flask, render_template, request, jsonify
import torch
import json
import pickle
from model_class import SignLanguageTranslationModel
import requests
import torch.nn.functional as F
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://43.203.16.219:8080"}})

model_path = "/home/ubuntu/modelServer/src/train_All_test_embedding_state.pth"

# 모델 객체 생성
pose_input_dim = 4  
hand_input_dim = 6
meaning_input_dim = 768
hidden_dim = 512
output_dim = 768
model = SignLanguageTranslationModel(pose_input_dim, hand_input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  

embedding_dict_path = "/home/ubuntu/modelServer/src/embedding_dict_new.pkl"
with open(embedding_dict_path, 'rb') as f:
  embedding_dict = pickle.load(f)

def preprocess_keypoints(keypoints, input_dim, target_num_keypoints):
    if not keypoints:  # 빈 배열일 경우
        return torch.zeros((1, target_num_keypoints, input_dim))  # 0으로 채운 텐서 반환

    x = [kp['x'] for kp in keypoints]
    y = [kp['y'] for kp in keypoints]
    z = [kp['z'] for kp in keypoints]

    if input_dim == 4:
        visibility = [kp.get('visibility', 1.0) for kp in keypoints]
        tensor = torch.tensor([x, y, z, visibility]).float().unsqueeze(0)  
    else:
        tensor = torch.tensor([x, y, z]).float().unsqueeze(0)  
    # 패딩 적용
    num_keypoints = tensor.size(2)
    if num_keypoints < target_num_keypoints:
        padding = torch.zeros((1, tensor.size(1), target_num_keypoints - num_keypoints)).float()
        tensor = torch.cat((tensor, padding), dim=2)
    elif num_keypoints > target_num_keypoints:
        tensor = tensor[:, :, :target_num_keypoints]  # 넘치는 부분 잘라냄

    return tensor.transpose(1, 2) 

#입력값 통한 의미 추론
def infer_meaning(model, pose_keypoints, left_hand_keypoints, right_hand_keypoints, embedding_dict):
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

    with torch.no_grad():
        outputs = model(pose_tensor, torch.cat((left_hand_tensor, right_hand_tensor), dim=2))

    model_embedding = outputs.squeeze(0)  # 모델 출력의 임베딩 벡터화
    
    # 코사인 유사도를 사용하여 가장 유사한 임베딩을 찾음
    best_similarity = -1  
    predicted_meaning = None

    # embedding_dict의 각 임베딩과 모델 출력 임베딩 간의 코사인 유사도 계산
    for meaning, embedding in embedding_dict.items():
        embedding_tensor = torch.tensor(embedding).to(device)
        similarity = F.cosine_similarity(model_embedding.unsqueeze(0), embedding_tensor.unsqueeze(0), dim=1)

        if similarity.item() > best_similarity:
            best_similarity = similarity.item()
            predicted_meaning = meaning

    return predicted_meaning

def load_json_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # HTTP 에러 발생 시 예외 발생
    return response.json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

#전달 받은 S3 URL을 통해 동작의 좌표를 추출한 json 파일 받아옴
def predict():
    data = request.get_json()
    json_url = data['s3url'].strip('\"')
    
    try:
        json_data = load_json_from_url(json_url)
    except Exception as e:
        return jsonify({'error': str(e)})

    predicted_meaning = infer_meaning(model, json_data["pose_keypoint"][0], json_data["left_hand_keypoint"][0], json_data["right_hand_keypoint"][0], embedding_dict)
    response = jsonify(predicted_meaning)
    response.headers.add('Content-Type', 'application/json; charset=utf-8')

    return app.response_class(
        response=json.dumps(predicted_meaning, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )
    

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)

