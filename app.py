from flask import Flask, render_template, request, jsonify
import torch
import json
import pickle
from model_class import SignLanguageTranslationModel

app = Flask(__name__)

model_path = r"C:\Users\User\OneDrive\바탕 화면\졸업프로젝트\modelServer\src\trainAll_state.pth"
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

meaning_dict_path = r"C:\Users\User\OneDrive\바탕 화면\졸업프로젝트\modelServer\src\meaning_dict.pkl"
with open(meaning_dict_path, 'rb') as f:
  meaning_dict = pickle.load(f)


def preprocess_keypoints(keypoints, input_dim):
    x = [kp['x'] for kp in keypoints]
    y = [kp['y'] for kp in keypoints]
    z = [kp['z'] for kp in keypoints]

    if input_dim == 4:
        visibility = [kp.get('visibility', 1.0) for kp in keypoints]
        tensor = torch.tensor([x, y, z, visibility]).float().unsqueeze(0)  # (1, 4, num_keypoints)
    else:
        tensor = torch.tensor([x, y, z]).float().unsqueeze(0)  # (1, 3, num_keypoints)

    return tensor.transpose(1, 2)  # (1, num_keypoints, input_dim)

def infer_meaning(model, pose_keypoints, left_hand_keypoints, right_hand_keypoints, meaning_dict):
    pose_tensor = preprocess_keypoints(pose_keypoints, input_dim=4)
    left_hand_tensor = preprocess_keypoints(left_hand_keypoints, input_dim=3)
    right_hand_tensor = preprocess_keypoints(right_hand_keypoints, input_dim=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    pose_tensor = pose_tensor.to(device)
    left_hand_tensor = left_hand_tensor.to(device)
    right_hand_tensor = right_hand_tensor.to(device)

    hand_inputs = torch.cat((left_hand_tensor, right_hand_tensor), dim=2)

    meaning_inputs = torch.zeros((1, 1, 5000)).to(device)

    with torch.no_grad():
        outputs = model(pose_tensor, hand_inputs, meaning_inputs)

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

#@app.route('/')
#def home():
#    return 'This is Home!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        if 'data' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['data']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        json_data = json.load(file)

        
        predicted_meaning = infer_meaning(model, json_data["pose_keypoint"][0], json_data["left_hand_keypoint"][0], json_data["right_hand_keypoint"][0], meaning_dict)
        response = jsonify({'predicted_meaning': predicted_meaning})
        response.headers.add('Content-Type', 'application/json; charset=utf-8')

        return response

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)

