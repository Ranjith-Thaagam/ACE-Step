import os
import pandas as pd
import torchaudio
import json
from datasets import Dataset, DatasetDict
import numpy as np
import gc

def clean_metadata(df):
    """Clean the metadata by removing rows with NaN values"""
    print("🧹 Cleaning metadata...")
    
    # Remove rows with NaN in critical columns
    initial_count = len(df)
    
    # Check which columns have NaN
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"⚠️  Found NaN values in columns: {nan_columns}")
    
    # Remove rows with NaN in path columns
    df_clean = df.dropna(subset=['audio_path', 'lyrics_path', 'prompt_path'])
    
    # Also remove rows where paths are not strings
    df_clean = df_clean[df_clean['audio_path'].apply(lambda x: isinstance(x, str))]
    df_clean = df_clean[df_clean['lyrics_path'].apply(lambda x: isinstance(x, str))]
    df_clean = df_clean[df_clean['prompt_path'].apply(lambda x: isinstance(x, str))]
    
    final_count = len(df_clean)
    print(f"📊 Cleaned metadata: {initial_count} -> {final_count} samples")
    
    return df_clean

def process_audio_safe(audio_path, target_sr=48000, max_duration=30):
    """Safely process audio with memory limits"""
    try:
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        
        # Limit duration to save memory (30 seconds max)
        max_samples = target_sr * max_duration
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
            print(f"⏰ Trimmed {audio_path} to {max_duration}s")
        
        # Convert to float32 and normalize
        waveform = waveform.float()
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max() * 0.9
            
        return waveform.numpy(), target_sr
        
    except Exception as e:
        print(f"❌ Error processing audio {audio_path}: {e}")
        return None, None

def convert_tamil_to_ace_step_format(metadata_path, songs_path, output_path, max_samples=None):
    """Convert your Tamil dataset to ACE Step format safely"""
    
    print("🔄 Converting Tamil dataset to ACE Step format...")
    
    # Load and clean metadata
    if metadata_path.endswith('.csv'):
        df = pd.read_csv(metadata_path)
    else:
        raise ValueError("Metadata should be CSV format")
    
    df_clean = clean_metadata(df)
    
    # Limit samples for testing if needed
    if max_samples:
        df_clean = df_clean.head(max_samples)
        print(f"🧪 Limiting to {max_samples} samples for testing")
    
    data_records = []
    processed_count = 0
    error_count = 0
    
    for idx, row in df_clean.iterrows():
        try:
            audio_path = os.path.join(songs_path, row['audio_path'])
            lyrics_path = os.path.join(songs_path, row['lyrics_path'])
            prompt_path = os.path.join(songs_path, row['prompt_path'])
            
            # Verify files exist
            if not all(os.path.exists(p) for p in [audio_path, lyrics_path, prompt_path]):
                missing = [p for p in [audio_path, lyrics_path, prompt_path] if not os.path.exists(p)]
                print(f"❌ Missing files: {missing}")
                error_count += 1
                continue
            
            # Process audio safely
            audio_array, sr = process_audio_safe(audio_path)
            if audio_array is None:
                error_count += 1
                continue
            
            # Load text files
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics = f.read().strip()
            
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            # Create record - store only audio path to save memory
            record = {
                "audio_path": audio_path,  # Store path instead of array to save memory
                "text": prompt,
                "lyrics": lyrics,
                "duration": audio_array.shape[1] / sr,
                "key": "C",  # Default key
                "sampling_rate": sr,
                "channels": audio_array.shape[0]
            }
            
            data_records.append(record)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"✅ Processed {processed_count}/{len(df_clean)} samples")
                gc.collect()  # Clean memory regularly
                
        except Exception as e:
            print(f"❌ Error processing row {idx}: {e}")
            error_count += 1
            continue
    
    print(f"📊 Processing complete: {processed_count} success, {error_count} errors")
    
    if processed_count == 0:
        print("❌ No samples processed successfully!")
        return None
    
    # Create Hugging Face Dataset
    try:
        dataset = Dataset.from_list(data_records)
        
        # Split into train/validation (90/10)
        if len(dataset) > 10:
            train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
            dataset_dict = DatasetDict({
                'train': train_test_split['train'],
                'validation': train_test_split['test']
            })
        else:
            # If too few samples, use all for training
            dataset_dict = DatasetDict({
                'train': dataset,
                'validation': dataset.select(range(min(2, len(dataset))))
            })
        
        # Save to disk
        dataset_dict.save_to_disk(output_path)
        
        print(f"✅ Dataset converted successfully!")
        print(f"📊 Train samples: {len(dataset_dict['train'])}")
        print(f"📊 Validation samples: {len(dataset_dict['validation'])}")
        print(f"💾 Saved to: {output_path}")
        
        return dataset_dict
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return None

def create_minimal_dataset_for_testing():
    """Create a very small dataset for immediate testing"""
    print("📝 Creating minimal Tamil dataset for testing...")
    
    records = []
    
    # Create just 8 samples for testing
    for i in range(8):
        # Very short audio to save memory (3 seconds)
        short_audio = np.random.randn(2, 48000 * 3).astype(np.float32) * 0.01
        
        record = {
            "audio_path": f"dummy_audio_{i}.wav",
            "text": [
                "இசையுடன் கூடிய தமிழ்ப்பாடல்",
                "மெல்லிசை தமிழ்ப்பாடல்",
                "காதல் பாடல் தமிழில்", 
                "பாடல் பாடும் இசை"
            ][i % 4],
            "lyrics": [
                "வணக்கம் வணக்கம்\nஇசையின் மகிழ்ச்சி",
                "பாடல் பாடுவேன்\nஇனிமையான இசை", 
                "தமிழிசை வாழ்க\nஎன்றும் வாழ்க",
                "மெல்லிசை கேட்போம்\nமகிழ்ச்சி பெறுவோம்"
            ][i % 4],
            "duration": 3.0,
            "key": ["C", "G", "D", "A"][i % 4],
            "sampling_rate": 48000,
            "channels": 2
        }
        records.append(record)
    
    # Create dataset
    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({
        'train': dataset,
        'validation': dataset.select(range(2))  # 2 samples for validation
    })
    
    # Save
    output_path = "data/tamil_dataset_minimal"
    dataset_dict.save_to_disk(output_path)
    
    print(f"✅ Minimal dataset created!")
    print(f"📊 Samples: {len(dataset)}")
    print(f"💾 Saved to: {output_path}")
    
    return dataset_dict

if __name__ == "__main__":
    print("Choose conversion method:")
    print("1. Full conversion (may take time)")
    print("2. Test conversion (first 20 samples)") 
    print("3. Minimal dataset (instant testing)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        convert_tamil_to_ace_step_format(
            metadata_path="data/tamil_data/metadata.csv",
            songs_path="data/tamil_data/songs/", 
            output_path="data/tamil_dataset_ace_format"
        )
    elif choice == "2":
        convert_tamil_to_ace_step_format(
            metadata_path="data/tamil_data/metadata.csv",
            songs_path="data/tamil_data/songs/", 
            output_path="data/tamil_dataset_test",
            max_samples=20
        )
    else:
        create_minimal_dataset_for_testing()
        