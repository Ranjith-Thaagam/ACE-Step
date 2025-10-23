from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
class DataLoader(Dataset):
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path

    def load_metadata(self):
        if self.metadata_path.endswith('.txt'):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            metadata = [line.strip().split('|') for line in lines]

        elif self.metadata_path.endswith('.csv'):
            df = pd.read_csv(self.metadata_path)
            metadata = df.values.tolist()
        else:
            raise ValueError("Unsupported metadata format (must be .txt or .csv)")

        return metadata

    def load_audio_data(self,path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate
    
    def load_lyrics_data(self,path):
        lyrics = None
        if path.endswith('.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                lyrics = f.read()

        elif path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(path)
            lyrics = df.to_dict(orient='records')

        return lyrics

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = row['audio_path']
        lyrics_path = row['lyrics_path']
        prompt_path = row['prompt_path']

        # --- Load audio ---
        if audio_path:
            waveform, sr = torchaudio.load(audio_path)
            if sr != 48000:
                waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
                sr = 48000
        else:
            waveform = torch.zeros(1, 48000)  # dummy 1-second silent audio
            sr = 48000

        # --- Load lyrics ---
        lyrics = ""
        if lyrics_path:
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read()

        # --- Load prompt ---
        prompt = ""
        if prompt_path:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read()

        return {
            "target_wavs": waveform,
            "wav_lengths": torch.tensor([waveform.shape[1]], dtype=torch.long),
            "prompts": prompt,
            "lyrics": lyrics,
        }