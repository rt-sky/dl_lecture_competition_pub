import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", target_fs: int = 120) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).numpy().copy()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # データの前処理
        self.X = self.preprocess_meg_data(self.X, target_fs)
        self.X = torch.tensor(self.X.copy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def preprocess_meg_data(self, data, target_fs=120):
        # ダウンサンプリング
        data = resample(data, int(data.shape[1] * target_fs / 1200), axis=1)
        
        # ベースライン補正
        baseline_start = 0  # エポックの開始
        baseline_end = int(target_fs * 0.5)  # 刺激の開始前500ms
        baseline = np.mean(data[:, :, baseline_start:baseline_end], axis=2, keepdims=True)
        data = data - baseline
        
        # センタリングとクリッピング
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        data = np.clip(data, -5, 5)

        # データの標準化
        # scaler = StandardScaler()
        # data = scaler.fit_transform(data.T).T
        
        # バンドパスフィルタの適用
        data = self.bandpass_filter(data, lowcut=0.1, highcut=30.0, fs=target_fs)
        
        return data

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        
        padlen = min(3 * max(len(b), len(a)), data.shape[1] - 1)
        
        return filtfilt(b, a, data, axis=1, padlen=padlen)
