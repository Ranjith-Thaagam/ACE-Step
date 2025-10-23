import os
import unicodedata
from src.data.loader import DataLoader
from src.tokenizer.tokenizer import TamilTokenizerWrapper, TrainTamilTokenizer

# ----------------------------
# Configuration
# ----------------------------
metadata_path = 'data/tamil_data/metadata.csv'
songs_path = 'data/tamil_data/songs/'
corpus_file = "tamil_corpus.txt"
model_output_dir = "chkpts/tokenizer/"
model_name = "tamil_tokenizer_special"

# ----------------------------
# Step 1: Build corpus from lyrics
# ----------------------------
def build_corpus(metadata):
    corpus = []

    for data in metadata:
        # Assuming lyrics path is the 2nd index (adjust if different)
        lyrics_file = data[2] if len(data) > 2 else None
        if lyrics_file and str(lyrics_file) != "nan":
            full_path = os.path.join(songs_path, lyrics_file)
            lyrics_text = data_loader.load_lyrics_data(full_path)
            if lyrics_text:  # Only add non-empty lyrics
                corpus.append(lyrics_text)

    # Join all songs with multiple newlines to separate them
    corpus_text = "\n\n".join(corpus)

    # Normalize Tamil text to NFC (important for combining letters)
    corpus_text = unicodedata.normalize('NFC', corpus_text)

    # Save corpus to file
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write(corpus_text)

    print(f"✅ Corpus written: {corpus_file}")
    print(f"Total unique chars: {len(set(corpus_text))}, Total length: {len(corpus_text)}")
    return corpus_text

# ----------------------------
# Main Training Process
# ----------------------------
if __name__ == "__main__":
    # Load or build corpus
    if not os.path.exists(corpus_file):
        print("📁 Building corpus from metadata...")
        data_loader = DataLoader(metadata_path=metadata_path)
        metadata = data_loader.load_metadata()
        corpus_text = build_corpus(metadata)
    else:
        print("📁 Loading existing corpus...")
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus_text = f.read()

    # Calculate vocabulary size
    num_chars = len(set(corpus_text))
    vocab_size = min(8192, max(512, num_chars * 2))  # More conservative calculation
    
    print(f"📊 Corpus stats:")
    print(f"  - Unique characters: {num_chars}")
    print(f"  - Total length: {len(corpus_text)}")
    print(f"  - Selected vocab size: {vocab_size}")

    # ----------------------------
    # Train tokenizer with special tokens
    # ----------------------------
    print("🔄 Training tokenizer with special tokens...")
    success = TrainTamilTokenizer.train(
        input_file=corpus_file,
        model_prefix=model_name,
        vocab_size=vocab_size,
        model_type='bpe'
    )

    if success:
        # ----------------------------
        # Test the trained tokenizer
        # ----------------------------
        print("\n🎯 Testing trained tokenizer...")
        tokenizer = TamilTokenizerWrapper(
            model_path=f'{model_output_dir}/{model_name}.model',
            device='cpu'
        )

        # Comprehensive test
        test_samples = [
            "வணக்கம்",
            "பாடல் பாடுவேன் இசையுடன்",
            "கலை எங்கள் வாழ்க்கை",
            "இசை உலகம் அழகானது",
            "பாடலில் இனிமை நிறைந்தது"
        ]

        print("\n🔤 Final Tokenizer Test:")
        for sample in test_samples:
            inputs = tokenizer(sample, return_tensors="pt", max_length=20)
            decoded = tokenizer.decode(inputs['input_ids'][0])
            
            print(f"Original: {sample}")
            print(f"Encoded: {inputs['input_ids'][0].tolist()}")
            print(f"Decoded: {decoded}")
            print(f"Match: {sample == decoded}")
            print("-" * 50)

        print("✅ Tamil tokenizer training completed successfully!")
        print(f"Model saved: {model_output_dir}/{model_name}.model")
    else:
        print("❌ Tokenizer training failed!")