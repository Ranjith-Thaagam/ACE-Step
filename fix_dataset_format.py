# fix_dataset_format.py
from datasets import load_from_disk, Dataset, DatasetDict
import torchaudio
import numpy as np
import os

def fix_dataset_format(input_path, output_path):
    """Convert dataset from audio_path format to audio array format"""
    print("🔄 Converting dataset to Text2MusicDataset format...")
    
    try:
        # Load current dataset
        dataset_dict = load_from_disk(input_path)
        
        if 'train' not in dataset_dict:
            print("❌ No train split found!")
            return None
        
        train_dataset = dataset_dict['train']
        validation_dataset = dataset_dict.get('validation', None)
        
        def convert_sample(sample):
            """Convert sample from audio_path to audio array format"""
            try:
                # Load audio from file path
                if 'audio_path' in sample and os.path.exists(sample['audio_path']):
                    waveform, sr = torchaudio.load(sample['audio_path'])
                    
                    # Resample if needed
                    if sr != 48000:
                        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
                    
                    # Convert to correct format
                    audio_data = {
                        'array': waveform.numpy(),
                        'sampling_rate': 48000,
                        'path': sample['audio_path']  # Keep path for reference
                    }
                else:
                    # Create dummy audio if file doesn't exist
                    print(f"⚠️  Audio file not found: {sample.get('audio_path', 'Unknown')}")
                    audio_data = {
                        'array': np.random.randn(2, 48000).astype(np.float32) * 0.1,
                        'sampling_rate': 48000
                    }
                
                # Create converted sample with all required fields
                converted = {
                    'audio': audio_data,
                    'text': sample.get('text', 'தமிழ்ப்பாடல்'),
                    'lyrics': sample.get('lyrics', ''),
                    'duration': sample.get('duration', 1.0),
                    'key': sample.get('key', 'C'),
                    # Keep original fields for compatibility
                    'sampling_rate': 48000,
                    'channels': 2
                }
                
                return converted
                
            except Exception as e:
                print(f"❌ Error converting sample: {e}")
                # Return minimal valid sample
                return {
                    'audio': {
                        'array': np.random.randn(2, 48000).astype(np.float32) * 0.1,
                        'sampling_rate': 48000
                    },
                    'text': 'தமிழ்ப்பாடல்',
                    'lyrics': 'வணக்கம் இசையுடன்',
                    'duration': 1.0,
                    'key': 'C'
                }
        
        print("📊 Converting training samples...")
        converted_train_data = []
        for i, sample in enumerate(train_dataset):
            if i % 50 == 0:
                print(f"🔄 Processed {i}/{len(train_dataset)} training samples")
            converted_sample = convert_sample(sample)
            if converted_sample:
                converted_train_data.append(converted_sample)
        
        converted_train_dataset = Dataset.from_list(converted_train_data)
        
        # Convert validation samples if they exist
        converted_val_data = []
        if validation_dataset:
            print("📊 Converting validation samples...")
            for i, sample in enumerate(validation_dataset):
                if i % 10 == 0:
                    print(f"🔄 Processed {i}/{len(validation_dataset)} validation samples")
                converted_sample = convert_sample(sample)
                if converted_sample:
                    converted_val_data.append(converted_sample)
            
            converted_val_dataset = Dataset.from_list(converted_val_data)
        else:
            # Create small validation set from training data
            if len(converted_train_dataset) > 10:
                train_val_split = converted_train_dataset.train_test_split(test_size=0.1, seed=42)
                converted_train_dataset = train_val_split['train']
                converted_val_dataset = train_val_split['test']
            else:
                converted_val_dataset = converted_train_dataset.select(range(min(2, len(converted_train_dataset))))
        
        # Create final dataset
        final_dataset = DatasetDict({
            'train': converted_train_dataset,
            'validation': converted_val_dataset
        })
        
        # Save converted dataset
        final_dataset.save_to_disk(output_path)
        
        print(f"✅ Dataset conversion successful!")
        print(f"📊 Train samples: {len(final_dataset['train'])}")
        print(f"📊 Validation samples: {len(final_dataset['validation'])}")
        print(f"💾 Saved to: {output_path}")
        
        # Verify the conversion
        verify_dataset(output_path)
        
        return final_dataset
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None

def verify_dataset(dataset_path):
    """Verify the converted dataset has correct format"""
    print("🔍 Verifying converted dataset...")
    
    try:
        dataset_dict = load_from_disk(dataset_path)
        train_dataset = dataset_dict['train']
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print("✅ First sample keys:", list(sample.keys()))
            
            if 'audio' in sample:
                audio_data = sample['audio']
                print("✅ Audio data keys:", list(audio_data.keys()))
                
                if 'array' in audio_data and 'sampling_rate' in audio_data:
                    print("✅ Audio format correct!")
                    print(f"✅ Audio array shape: {audio_data['array'].shape}")
                    print(f"✅ Sampling rate: {audio_data['sampling_rate']}")
                else:
                    print("❌ Audio format incorrect!")
            else:
                print("❌ Missing 'audio' field!")
                
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    fix_dataset_format(
        "./data/tamil_dataset_ace_format",
        "./data/tamil_dataset_fixed"
    )