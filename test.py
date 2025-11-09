class AudioDataset(Dataset):

    def __init__(self, dataframe, audio_dir, processor, max_length=40):
        self.df = dataframe.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = str(row['file_name'])
        transcription = row['transcription']
        
        if not file_name.endswith('.wav'):
            return self._get_dummy_sample()
        
        try:
            audio, sr = librosa.load(os.path.join(self.audio_dir, file_name), sr=16000)
            max_samples = int(self.max_length * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest"
            )

            labels = self.processor.tokenizer(
                transcription,
                return_tensors="pt",
                padding="longest"
            )
            
            label_ids = labels.input_ids.squeeze(0)
            # max_token_id = label_ids.max().item()
            # vocab_size = len(self.processor.tokenizer)

            # unk_token_id = self.processor.tokenizer.unk_token_id
            # if unk_token_id in label_ids:
            #     unk_positions = [i for i, tid in enumerate(label_ids) if tid == unk_token_id]

            #     print(f"ERROR unk_token in {file_name}")
            #     print(f"   Text: '{transcription}'")
            #     print(f"   Max token ID: {max_token_id}, Vocab size: {vocab_size}")
            #     print(f"   Token IDs: {label_ids.tolist()}")
            #     print(f"   Tokens: {self.processor.tokenizer.convert_ids_to_tokens(label_ids)}")
            #     print(f"   unk_positions: {unk_positions}")
            #     # exit()
            # if max_token_id >= vocab_size:
            #     print(f"ERROR max_token_id > vocab_size in {file_name}")
            #     print(f"   Text: '{transcription}'")
            #     print(f"   Max token ID: {max_token_id}, Vocab size: {vocab_size}")
            #     print(f"   Token IDs: {label_ids.tolist()}")
            #     print(f"   Tokens: {self.processor.tokenizer.convert_ids_to_tokens(label_ids)}")
            #     # Return dummy sample instead of crashing
            #     # exit()
            #     return self._get_dummy_sample()

            return {
                "input_features": inputs.input_features.squeeze(0),
                "labels": label_ids
            }

        except Exception as e:
            print(f"error: {file_name}: {e}")
            return {
                "input_features": torch.zeros(80, 3000),
                "labels": torch.zeros(1, dtype=torch.long)
            }

def generation_test(self):
    """Test model generation on entire train and validation datasets - one sample at a time"""
    print("\n" + "="*70)
    print("GENERATION TEST - Testing all samples (one at a time)")
    print("="*70)
    
    # Create data collator for padding single samples
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=self.processor,
    )
    
    vocab_size = len(self.processor.tokenizer)
    device = self.model.device
    
    # Test both splits
    for split_name, dataset in [("Train", self.train_dataset), ("Validation", self.val_dataset)]:
        print(f"Testing {split_name} Dataset ({len(dataset)} samples)")
        
        issues = []
        total_samples = len(dataset)
        
        for idx in range(total_samples):
            print(f"Testing {split_name} Dataset, sample number {idx}")
            # Get single sample
            sample = dataset[idx]
            file_name = dataset.df.iloc[idx]['file_name']
            transcription = dataset.df.iloc[idx]['transcription']

            # Collate single sample (wrap in list, returns batch of 1)
            try:
                batch = data_collator([sample])
            except Exception as e:
                print(f"\n❌ Error collating sample {idx}: {e}")
                issues.append({
                    'split': split_name,
                    'idx': idx,
                    'file': file_name,
                    'text': transcription,
                    'error': f'Collation error: {e}'
                })
                continue
            
            # Move to device
            input_features = batch["input_features"].to(device)  # [1, 80, 3000]
            labels = batch["labels"]  # [1, seq_len]
            
            label_ids = labels.squeeze(0)
            unk_positions = [i for i, tid in enumerate(label_ids) if tid == self.processor.tokenizer.unk_token_id]
            print("> sample info:")
            print(f"> File name: {file_name}")
            print(f"> Text: '{transcription}'")
            print(f"> Max token ID: {label_ids.max().item()}, Vocab size: {vocab_size}")
            print(f"> Token IDs: {label_ids.tolist()}")
            print(f"> Tokens: {self.processor.tokenizer.convert_ids_to_tokens(label_ids)}")
            print(f"> unk_positions: {unk_positions}")
            
            # Check labels for invalid IDs BEFORE generation
            label = labels[0]  # Get the single sample's labels
            label_no_padding = label[label != -100]
            if len(label_no_padding) > 0:
                max_label_id = label_no_padding.max().item()
                min_label_id = label_no_padding.min().item()
                
                if max_label_id >= vocab_size:
                    print(f"\n❌ Sample {idx}: Label token ID {max_label_id} >= vocab size {vocab_size}")
                    issues.append({
                        'split': split_name,
                        'idx': idx,
                        'file': file_name,
                        'text': transcription,
                        'error': f'Label token ID {max_label_id} >= vocab size {vocab_size}',
                        'label_ids': label_no_padding.tolist(),
                        'tokens': self.processor.tokenizer.convert_ids_to_tokens(label_no_padding)
                    })
                    continue  # Skip generation if labels are invalid
                
                if min_label_id < 0 and min_label_id != -100:
                    print(f"\n❌ Sample {idx}: Invalid negative label token ID {min_label_id}")
                    issues.append({
                        'split': split_name,
                        'idx': idx,
                        'file': file_name,
                        'text': transcription,
                        'error': f'Invalid negative token ID {min_label_id}',
                        'label_ids': label_no_padding.tolist()
                    })
                    continue
            
            # Generate
            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_features,
                        max_new_tokens=225,
                        num_beams=1,
                        do_sample=False,
                        decoder_start_token_id=self.processor.tokenizer.bos_token_id,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )  # Returns [1, generated_length]
                
                # Check for invalid token IDs in generation
                generated_ids_flat = generated_ids[0]  # Get the single sample
                max_generated_id = generated_ids_flat.max().item()
                min_generated_id = generated_ids_flat.min().item()
                
                if max_generated_id >= vocab_size:
                    print(f"\n❌ Sample {idx}: Generated token ID {max_generated_id} >= vocab size {vocab_size}")
                    issues.append({
                        'split': split_name,
                        'idx': idx,
                        'file': file_name,
                        'text': transcription,
                        'error': f'Generated token ID {max_generated_id} >= vocab size {vocab_size}',
                        'generated_ids': generated_ids_flat.tolist(),
                        'tokens': self.processor.tokenizer.convert_ids_to_tokens(generated_ids_flat)
                    })
                    continue
                
                if min_generated_id < 0:
                    print(f"\n❌ Sample {idx}: Invalid negative generated token ID {min_generated_id}")
                    issues.append({
                        'split': split_name,
                        'idx': idx,
                        'file': file_name,
                        'text': transcription,
                        'error': f'Invalid negative generated token ID {min_generated_id}',
                        'generated_ids': generated_ids_flat.tolist()
                    })
                
            except Exception as e:
                print(f"\n❌ Sample {idx}: Generation error: {e}")
                issues.append({
                    'split': split_name,
                    'idx': idx,
                    'file': file_name,
                    'text': transcription,
                    'error': f'Generation error: {e}'
                })
            
            # Progress indicator (update every 50 samples)
            if (idx + 1) % 50 == 0 or (idx + 1) == total_samples:
                print(f"  Processed {idx + 1}/{total_samples} samples...", end='\r')
        
        print(f"  Processed {total_samples}/{total_samples} samples - Complete!")
        
        # Report issues for this split
        split_issues = [iss for iss in issues if iss['split'] == split_name]
        if split_issues:
            print(f"\n  ⚠️  Found {len(split_issues)} issues in {split_name}")
        else:
            print(f"  ✅ No issues found in {split_name}")
    
    # Summary
    print(f"\n{'='*70}")
    print("GENERATION TEST SUMMARY")
    print(f"{'='*70}")
    
    if issues:
        print(f"❌ Found {len(issues)} total issues!")
        print(f"\nShowing first 10 issues:\n")
        
        for i, issue in enumerate(issues[:10], 1):
            print(f"{i}. {issue['split']} - Index {issue['idx']}")
            print(f"   File: {issue['file']}")
            print(f"   Text: '{issue['text']}'")
            print(f"   Error: {issue['error']}")
            
            if 'tokens' in issue:
                print(f"   Tokens: {issue['tokens'][:20]}...")
            
            if 'generated_ids' in issue:
                ids_preview = issue['generated_ids'][:20]
                print(f"   Generated IDs: {ids_preview}{'...' if len(issue['generated_ids']) > 20 else ''}")
            
            if 'label_ids' in issue:
                ids_preview = issue['label_ids'][:20]
                print(f"   Label IDs: {ids_preview}{'...' if len(issue['label_ids']) > 20 else ''}")
            
            print()
        
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more issues")
        
        print(f"\n{'='*70}")
        raise ValueError(f"Generation test failed with {len(issues)} issues!")
    else:
        print("✅ ALL TESTS PASSED!")
        print(f"   - Train samples tested: {len(self.train_dataset)}")
        print(f"   - Validation samples tested: {len(self.val_dataset)}")
        print(f"   - Total samples tested: {len(self.train_dataset) + len(self.val_dataset)}")
        print(f"   - Vocab size: {vocab_size}")
        print(f"   - All generated token IDs within valid range [0, {vocab_size-1}]")
        print(f"   - All label token IDs within valid range [0, {vocab_size-1}]")
        print(f"{'='*70}\n")