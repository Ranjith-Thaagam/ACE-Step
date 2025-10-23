import torch
import sentencepiece as spm
import os

class TamilTokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Define special tokens mapping
        self.special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]', 
            'eos_token': '[EOS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]',
            'cls_token': '[CLS]'
        }
        
        # Map to actual token IDs
        self.pad_token_id = self.sp.piece_to_id('[PAD]') if self.sp.piece_to_id('[PAD]') != 0 else 0
        self.unk_token_id = self.sp.piece_to_id('[UNK]') if self.sp.piece_to_id('[UNK]') != 0 else 1
        self.bos_token_id = self.sp.piece_to_id('[BOS]') if self.sp.piece_to_id('[BOS]') != 0 else 2
        self.eos_token_id = self.sp.piece_to_id('[EOS]') if self.sp.piece_to_id('[EOS]') != 0 else 3

    def encode(self, text: str):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list):
        return self.sp.decode(ids)
    
    def get_vocab_size(self):
        return self.sp.get_piece_size()
    
    def encode_batch(self, texts: list, max_length: int = None, padding: bool = False, truncation: bool = False):
        """Encode a batch of texts with optional padding and truncation"""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = []
        for text in texts:
            tokens = self.encode(text)
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            encoded.append(tokens)
        
        if padding and max_length:
            # Pad sequences to max_length
            padded = []
            for seq in encoded:
                if len(seq) < max_length:
                    padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
                else:
                    padded_seq = seq
                padded.append(padded_seq)
            encoded = padded
            
        return encoded

class TamilTokenizerWrapper:
    def __init__(self, model_path: str, device: torch.device = torch.device('cpu')):
        self.tokenizer = TamilTokenizer(model_path)
        self.device = device
        
        # ACE Step expected attributes
        self.pad_token_id = self.tokenizer.pad_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def __call__(self, texts, return_tensors: str = "pt", padding: bool = True, 
                 truncation: bool = True, max_length: int = 256, **kwargs):
        """ACE Step compatible interface"""
        
        # Encode batch
        input_ids = self.tokenizer.encode_batch(
            texts, 
            max_length=max_length,
            padding=padding,
            truncation=truncation
        )
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).long()
        
        if return_tensors == "pt":
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            return {
                'input_ids': input_ids.cpu().numpy(),
                'attention_mask': attention_mask.cpu().numpy()
            }
    
    def encode(self, text: str):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()
        elif isinstance(ids, list) and all(isinstance(x, torch.Tensor) for x in ids):
            ids = [x.cpu().tolist() for x in ids]
        return self.tokenizer.decode(ids)
    
    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

class TrainTamilTokenizer:
    @staticmethod
    def train(input_file: str, model_prefix: str, vocab_size: int = 8000, model_type: str = 'bpe'):
        model_dir = "chkpts/tokenizer/"
        os.makedirs(model_dir, exist_ok=True)
        
        full_model_path = os.path.join(model_dir, model_prefix)
        
        # FIXED: Use only user_defined_symbols, not both control_symbols and user_defined_symbols
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=full_model_path,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            # Special tokens configuration - FIXED: Only use user_defined_symbols
            pad_id=0,
            pad_piece='[PAD]',
            unk_id=1, 
            unk_piece='[UNK]',
            bos_id=2,
            bos_piece='[BOS]',
            eos_id=3,
            eos_piece='[EOS]',
            # Additional symbols - FIXED: Only in user_defined_symbols
            user_defined_symbols=['[SEP]', '[MASK]', '[CLS]'],
            # Tamil-specific settings
            split_by_unicode_script=True,
            split_by_whitespace=True,
            split_by_number=True,
            treat_whitespace_as_suffix=False,
            # Training parameters
            num_threads=os.cpu_count(),
            max_sentence_length=16384,
            shuffle_input_sentence=True,
            # Additional settings for better Tamil handling
            byte_fallback=True,  # Helps with rare characters
            remove_extra_whitespaces=False,  # Keep original formatting
            add_dummy_prefix=False  # Don't add space at beginning
        )
        print(f"✅ Tokenizer trained and saved: {full_model_path}.model")
        
        # Verify the trained model
        return TrainTamilTokenizer.verify_tokenizer(full_model_path + '.model')
    
    @staticmethod
    def verify_tokenizer(model_path: str):
        """Verify the trained tokenizer works correctly with special tokens"""
        try:
            tokenizer = TamilTokenizer(model_path)
            
            print("🔍 Tokenizer Verification:")
            print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
            
            # Check special tokens
            special_tokens_check = {
                '[PAD]': tokenizer.sp.piece_to_id('[PAD]'),
                '[UNK]': tokenizer.sp.piece_to_id('[UNK]'),
                '[BOS]': tokenizer.sp.piece_to_id('[BOS]'),
                '[EOS]': tokenizer.sp.piece_to_id('[EOS]'),
                '[SEP]': tokenizer.sp.piece_to_id('[SEP]'),
                '[MASK]': tokenizer.sp.piece_to_id('[MASK]'),
                '[CLS]': tokenizer.sp.piece_to_id('[CLS]')
            }
            
            print("Special tokens mapping:")
            for token, token_id in special_tokens_check.items():
                print(f"  {token}: {token_id}")
            
            # Test with Tamil text
            test_texts = [
                "வணக்கம்",
                "பாடல் பாடுவேன்",
                "இசை மிகவும் அருமை",
                "கலை எங்கள் வாழ்க்கை"
            ]
            
            print("\n🧪 Test encoding/decoding:")
            for text in test_texts:
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                print(f"  '{text}' -> {encoded} -> '{decoded}'")
                assert text == decoded, f"Round-trip failed for: {text}"
            
            print("✅ All tests passed! Tokenizer is ready.")
            return True
            
        except Exception as e:
            print(f"❌ Tokenizer verification failed: {e}")
            return False