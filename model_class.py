import torch
import torch.nn as nn

class SignLanguageTranslationModel(nn.Module):
    def __init__(self, pose_input_dim, hand_input_dim, hidden_dim, output_dim):
        super(SignLanguageTranslationModel, self).__init__()
        self.pose_lstm = nn.LSTM(input_size=pose_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.hand_lstm = nn.LSTM(input_size=hand_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 for pose and hand

    def forward(self, pose_inputs, hand_inputs):
        pose_features, _ = self.pose_lstm(pose_inputs)
        hand_features, _ = self.hand_lstm(hand_inputs)
        combined_features = torch.cat((pose_features[:, -1], hand_features[:, -1]), dim=1)  # Combine final LSTM outputs
        outputs = self.fc(combined_features)
        return outputs

