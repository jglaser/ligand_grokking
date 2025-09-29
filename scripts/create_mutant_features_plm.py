"""
This script creates "delta vector" features for protein mutations using a
Hugging Face Protein Language Model (e.g., ProtBERT, ProtT5).

This version has been updated for methodological consistency with the AlphaGenome
featurization pipeline, ensuring robust validation of every mutation.

Workflow:
1.  Reads Parquet files containing `uniprot_id` and `mutation` columns.
2.  Groups all mutations by their unique UniProt ID.
3.  For each UniProt ID, it fetches the canonical protein sequence once.
4.  For each associated mutation string (e.g., 'L99A,V600E'), it splits the
    string and applies each point mutation sequentially to generate a final
    in-silico mutant sequence.
5.  Each point mutation undergoes a strict sanity check to ensure the
    reference amino acid at the specified position in the sequence matches.
6.  It uses the `HuggingFaceFeaturizer` to embed all validated wild-type
    and mutant sequences in batches, leveraging TPU/GPU acceleration.
7.  It calculates the "delta vector" (mutant_embedding - wild_type_embedding).
8.  The resulting delta vectors are saved to a compressed NumPy file (`.npz`).
"""
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import re
from bioservices import UniProt
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import warnings

# Suppress verbose warnings from the transformers library
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- XLA / MPS Backend Detection for Acceleration ---
def get_best_device():
    """Checks for and returns the best available device (TPU, MPS, CUDA, CPU)."""
    if 'XLA_USE_BF16' not in os.environ and 'XLA_USE_FP16' not in os.environ:
        os.environ['XLA_USE_BF16'] = '1'
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

class HuggingFaceFeaturizer:
    """A class to handle model loading and feature generation."""
    def __init__(self, device):
        self.device = device
        self.models = {}
        self.tokenizers = {}
        print(f"HuggingFaceFeaturizer initialized on device: {self.device}")

    def _load_model_and_tokenizer(self, model_name):
        """Loads a model and tokenizer if not already cached."""
        if model_name not in self.models:
            print(f"\nLoading model '{model_name}'...")
            if "t5" in model_name.lower():
                self.tokenizers[model_name] = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
                self.models[model_name] = T5EncoderModel.from_pretrained(model_name).to(self.device)
            else:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
                self.models[model_name] = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Use torch.compile for significant speedup if available
            if hasattr(torch, 'compile'):
                self.models[model_name] = torch.compile(self.models[model_name])
                print("Model compilation enabled.")
        return self.models[model_name], self.tokenizers[model_name]

    def featurize(self, sequences, model_name, batch_size):
        """Generates embeddings for a list of sequences."""
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        
        all_embeddings = []
        # Process sequences in batches to manage memory
        for i in tqdm(range(0, len(sequences), batch_size), desc=f"Featurizing with {model_name}"):
            batch_seqs = sequences[i:i + batch_size]
            # Add spaces between residues for ProtBERT/T5 tokenization style
            spaced_seqs = [" ".join(list(seq)) for seq in batch_seqs]
            
            inputs = tokenizer(spaced_seqs, add_special_tokens=True, padding="longest", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Extract last hidden state and average over sequence length
            embeddings = outputs.last_hidden_state.cpu().numpy()
            attention_mask = inputs.attention_mask.cpu().numpy()
            
            # Mask out padding tokens before averaging
            masked_embeddings = embeddings * np.expand_dims(attention_mask, axis=-1)
            summed_embeddings = np.sum(masked_embeddings, axis=1)
            seq_lens = np.sum(attention_mask, axis=1, keepdims=True)
            
            # Handle potential division by zero for empty sequences
            seq_lens[seq_lens == 0] = 1 
            
            avg_embeddings = summed_embeddings / seq_lens
            all_embeddings.append(avg_embeddings)
            
        return np.concatenate(all_embeddings, axis=0)

def main(args):
    print("--- Starting PLM Delta Vector Featurization Pipeline ---")
    
    device = get_best_device()
    featurizer = HuggingFaceFeaturizer(device)
    u_service = UniProt()

    df = pd.concat([pd.read_parquet(p) for p in args.input_paths])
    uniprot_to_mutations = df.groupby('uniprot_id')['mutation'].unique().to_dict()
    unique_uniprots = list(uniprot_to_mutations.keys())

    wt_sequences = {}
    mutant_sequences = {}
    
    print("Fetching sequences and validating mutations...")
    for uniprot_id in tqdm(unique_uniprots, desc="Processing Proteins"):
        try:
            fasta_data = u_service.retrieve(uniprot_id, "fasta")
            if not isinstance(fasta_data, str):
                tqdm.write(f"Warning: Could not retrieve FASTA for {uniprot_id}. Skipping.")
                continue
            
            wt_seq = "".join(fasta_data.splitlines()[1:])
            wt_sequences[uniprot_id] = wt_seq
            
            for mutation_str in uniprot_to_mutations.get(uniprot_id, []):
                mutant_seq_list = list(wt_seq)
                valid_mutation = True
                
                # Split mutation string and apply sequentially
                point_mutations = mutation_str.split(',')
                
                for point_mutation in point_mutations:
                    match = re.match(r'([A-Z])(\d+)([A-Z])', point_mutation)
                    if not match:
                        tqdm.write(f"Warning: Could not parse mutation format '{point_mutation}' in '{mutation_str}' for {uniprot_id}. Skipping entire set.")
                        valid_mutation = False
                        break
                    
                    ref_aa, pos_str, alt_aa = match.groups()
                    pos_idx = int(pos_str) - 1

                    # --- STRICT SANITY CHECK ---
                    # Use the current state of mutant_seq_list for validation
                    current_seq_for_validation = "".join(mutant_seq_list)
                    if not (0 <= pos_idx < len(current_seq_for_validation)):
                        tqdm.write(f"Warning: Position {pos_str} is out of bounds for {uniprot_id} (length {len(current_seq_for_validation)}). Skipping {mutation_str}.")
                        valid_mutation = False
                        break
                    
                    if current_seq_for_validation[pos_idx] != ref_aa:
                        tqdm.write(f"Warning: Reference mismatch for {point_mutation} in {uniprot_id}. Expected {ref_aa}, found {current_seq_for_validation[pos_idx]}. Skipping {mutation_str}.")
                        valid_mutation = False
                        break

                    # If the check passes, apply the mutation
                    mutant_seq_list[pos_idx] = alt_aa

                # If all point mutations in the set were valid, store the final sequence
                if valid_mutation:
                    mutant_sequences[f"{uniprot_id}_{mutation_str}"] = "".join(mutant_seq_list)

        except Exception as e:
            tqdm.write(f"An unexpected error occurred processing {uniprot_id}: {e}")

    # --- Batch Featurization ---
    wt_ids = list(wt_sequences.keys())
    wt_seqs_list = list(wt_sequences.values())
    mutant_target_ids = list(mutant_sequences.keys())
    mutant_seqs_list = list(mutant_sequences.values())

    print(f"\nFound {len(wt_sequences)} valid wild-type sequences.")
    print(f"Generated {len(mutant_sequences)} valid mutant sequences for featurization.")

    if not mutant_sequences:
        print("--- No valid mutant sequences generated. Exiting. ---")
        return

    wt_embeddings = featurizer.featurize(wt_seqs_list, args.model_name, args.batch_size)
    wt_embedding_map = {uid: emb for uid, emb in zip(wt_ids, wt_embeddings)}

    mutant_embeddings = featurizer.featurize(mutant_seqs_list, args.model_name, args.batch_size)
    mutant_embedding_map = {tid: emb for tid, emb in zip(mutant_target_ids, mutant_embeddings)}

    delta_vectors = {}
    print("\nCalculating delta vectors...")
    for target_id, mutant_emb in mutant_embedding_map.items():
        uniprot_id = target_id.split('_')[0]
        if uniprot_id in wt_embedding_map:
            wt_emb = wt_embedding_map[uniprot_id]
            delta_vectors[target_id] = mutant_emb # - wt_emb
            
    print(f"\nSuccessfully generated {len(delta_vectors)} delta vectors.")
    if len(delta_vectors) > 0:
      print(f"Saving vectors to '{args.output_path}'...")
      np.savez_compressed(args.output_path, **delta_vectors)
      print("--- Done ---")
    else:
      print("--- No vectors generated. Exiting. ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate delta vectors for mutant proteins using a Protein Language Model and alignment validation.")
    parser.add_argument('--input_paths', nargs='+', required=True, help='Path(s) to the Parquet split files.')
    parser.add_argument('--output_path', type=str, default='mutant_delta_vectors_plm_aligned.npz')
    parser.add_argument('--model_name', type=str, default='Rostlab/prot_t5_xl_uniref50', help='Name of the Hugging Face model to use.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for GPU/TPU processing.')
    args = parser.parse_args()
    main(args)
