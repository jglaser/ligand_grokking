"""
This script implements the core "delta vector" featurization pipeline using the
efficient `predict_variant` method from the AlphaGenome API.

Workflow:
1.  Identifies all unique (uniprot_id, mutation) pairs from the input data.
2.  **Processes each unique UniProt ID in parallel to speed up execution.**
3.  For each protein, it finds the best-matching Ensembl transcript by aligning
    against the UniProt sequence.
4.  For each mutation, it uses this best-match transcript to determine the precise
    genomic location and the required nucleotide change.
5.  It makes a single, efficient call to `ag_model_client.predict_variant()`.
6.  It creates the final "delta vector" (mutant - wild_type).
"""
import argparse
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import re
import time
from bioservices import Ensembl, UniProt
import warnings

# --- Bioinformatics Imports ---
try:
    from Bio.Align import PairwiseAligner
    from Bio import BiopythonDeprecationWarning
    warnings.simplefilter('ignore', BiopythonDeprecationWarning)
except ImportError:
    print("Biopython not found. Please install it: pip install biopython")
    exit()

# --- Concurrency ---
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- AlphaGenome Imports ---
from alphagenome.models import dna_client
from alphagenome.data import genome
from alphagenome.models import dna_output
from alphagenome.data import track_data


# A standard codon table for translating amino acids to DNA
CODON_TABLE = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'], 'D': ['GAT', 'GAC'], 'C': ['TGT', 'TGC'], 'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'], 'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'], 'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
    'K': ['AAA', 'AAG'], 'M': ['ATG'], 'F': ['TTT', 'TTC'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'], 'Y': ['TAT', 'TAC'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TAG', 'TGA']
}
# Create a reverse mapping from codon to amino acid for validation
CODON_TO_AA = {codon: aa for aa, codons in CODON_TABLE.items() for codon in codons}


def get_ensembl_gene_id(u_service, uniprot_id):
    """Maps a UniProt ID to a stable Ensembl Gene ID with retries."""
    for attempt in range(3):
        try:
            mapping_result = u_service.mapping(fr="UniProtKB_AC-ID", to="Ensembl", query=uniprot_id)
            if isinstance(mapping_result, dict) and mapping_result.get('results'):
                for entry in mapping_result['results']:
                    if entry.get('to', '').startswith('ENSG'):
                        return entry['to'].split('.')[0]
            time.sleep(1 * (attempt + 1)) # Wait before retrying
        except Exception:
            time.sleep(1 * (attempt + 1))
    return None

def find_best_matching_transcript(e_service, gene_data, uniprot_seq, max_transcripts_to_check):
    """
    Iterates through a prioritized subset of transcripts for a gene, aligns them
    to the UniProt sequence, and returns the one with the best alignment score.
    """
    transcripts = gene_data.get('Transcript', [])
    if not isinstance(transcripts, list) or not transcripts:
        return None, "Gene data contains no 'Transcript' list."

    protein_coding_transcripts = [t for t in transcripts if isinstance(t, dict) and t.get('biotype') == 'protein_coding']
    protein_coding_transcripts.sort(key=lambda t: t.get('end', 0) - t.get('start', 0), reverse=True)
    
    transcripts_to_check = protein_coding_transcripts[:max_transcripts_to_check]
    if len(protein_coding_transcripts) > max_transcripts_to_check:
        tqdm.write(f"  - Info: Gene {gene_data['id']} has {len(protein_coding_transcripts)} protein-coding transcripts. Checking the {max_transcripts_to_check} longest.")

    best_transcript = None
    best_score = -1

    aligner = PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    for t in transcripts_to_check:
        transcript_id = t.get('id')
        if not isinstance(transcript_id, str): continue
        
        full_transcript_data = e_service.get_lookup_by_id(transcript_id, expand=1)
        if not isinstance(full_transcript_data, dict): continue
        
        cdna_data = e_service.get_sequence_by_id(transcript_id, type='cdna')
        sequence = None
        if isinstance(cdna_data, str) and cdna_data.startswith('>'):
            sequence = "".join(cdna_data.splitlines()[1:])
        elif isinstance(cdna_data, dict) and 'seq' in cdna_data:
            sequence = cdna_data['seq']

        if isinstance(sequence, str):
            full_transcript_data['cdna'] = sequence
            ensembl_protein_seq = "".join([CODON_TO_AA.get(sequence[i:i+3], '?') for i in range(0, len(sequence), 3)])
            
            alignments = aligner.align(ensembl_protein_seq, uniprot_seq)
            if len(alignments) > 0:
                score = alignments[0].score
                if score > best_score:
                    best_score = score
                    best_transcript = full_transcript_data
    
    # Heuristic: score should be at least 80% of a perfect match to be considered valid
    if best_score < 0.8 * len(uniprot_seq) * 5:
        return None, f"No transcript found with a high-quality alignment (best score: {best_score})."

    return best_transcript, None


def get_variant_info(uniprot_protein_seq, transcript_data, mutation, gene_strand):
    """
    Translates a protein mutation to a genomic variant, using a pre-aligned transcript.
    """
    try:
        match = re.match(r'([A-Z])(\d+)([A-Z])', mutation)
        if not match: return None, f"Could not parse mutation format '{mutation}'."
        ref_aa, pos, alt_aa = match.groups()
        pos = int(pos)

        if 'cdna' not in transcript_data or 'Exon' not in transcript_data:
            return None, "Transcript data is missing 'cdna' or 'Exon' information."
 
        cds = transcript_data['cdna']
        ensembl_protein_seq = "".join([CODON_TO_AA.get(cds[i:i+3], '?') for i in range(0, len(cds), 3)])
        
        pos_idx = pos - 1
        if not (0 <= pos_idx < len(uniprot_protein_seq)):
            return None, f"Position {pos} is out of bounds for UniProt sequence."
        
        if uniprot_protein_seq[pos_idx] != ref_aa:
            return None, f"Mutation reference '{ref_aa}' does not match UniProt sequence '{uniprot_protein_seq[pos_idx]}' at position {pos}."
 
        # --- FIX: Correctly access the start coordinate from the alignment tuple ---
        aligner = PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 5
        aligner.mismatch_score = -4
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1
        alignments = aligner.align(ensembl_protein_seq, uniprot_protein_seq)
        if len(alignments) == 0: return None, "Alignment failed."

  
        offset = alignments[0].aligned[0][0][0]
        corrected_pos = pos + offset
        corrected_pos_idx = corrected_pos - 1

        if not (0 <= corrected_pos_idx < len(ensembl_protein_seq)):
            return None, f"Position {pos} (corrected to {corrected_pos}) is out of bounds for Ensembl protein."

        found_aa = ensembl_protein_seq[corrected_pos_idx]
        if found_aa != ref_aa:
            return None, f"Reference AA mismatch at aligned position {corrected_pos}. Expected {ref_aa}, but transcript has {found_aa}."

        codon_start_in_cds = (corrected_pos - 1) * 3
        original_codon = cds[codon_start_in_cds : codon_start_in_cds + 3]

        new_codon = CODON_TABLE.get(alt_aa, [None])[0]
        if new_codon is None: return None, f"Invalid alternate amino acid '{alt_aa}'."
        
        diff_idx = next((i for i, (c1, c2) in enumerate(zip(original_codon, new_codon)) if c1 != c2), -1)
        if diff_idx == -1: return None, "Mutation is synonymous (no nucleotide change)."

        cds_exons = sorted(transcript_data['Exon'], key=lambda x: x['start'])
        
        genomic_pos_of_mutation = -1
        if gene_strand == 1:
            bases_covered = 0
            for exon in cds_exons:
                exon_len = exon['end'] - exon['start'] + 1
                if bases_covered + exon_len > codon_start_in_cds + diff_idx:
                    offset_in_exon = (codon_start_in_cds + diff_idx) - bases_covered
                    genomic_pos_of_mutation = exon['start'] + offset_in_exon
                    break
                bases_covered += exon_len
        else: # Negative strand
            bases_covered = 0
            for exon in sorted(transcript_data['Exon'], key=lambda x: x['start'], reverse=True):
                exon_len = exon['end'] - exon['start'] + 1
                if bases_covered + exon_len > codon_start_in_cds + diff_idx:
                    offset_in_exon = (codon_start_in_cds + diff_idx) - bases_covered
                    genomic_pos_of_mutation = exon['end'] - offset_in_exon
                    break
                bases_covered += exon_len

        if genomic_pos_of_mutation == -1:
            return None, "Failed to map CDS position to a genomic coordinate."
        
        result = {
            "position": genomic_pos_of_mutation,
            "reference_bases": original_codon[diff_idx],
            "alternate_bases": new_codon[diff_idx]
        }
        return result, None
    except Exception as e:
        # Catch any unexpected errors during this complex process and report them
        return None, f"An unexpected exception occurred during variant mapping: {e} (Line: {e.__traceback__.tb_lineno})"


def flatten_predictions_to_vector(predictions: dna_output.Output) -> np.ndarray:
    """Processes a DNAOutput object into a single, flat numpy vector."""
    all_values = []
    for modality in sorted([m for m in dir(predictions) if isinstance(getattr(predictions, m), track_data.TrackData)]):
        track_data_obj = getattr(predictions, modality)
        if hasattr(track_data_obj, 'values') and track_data_obj.values is not None:
            all_values.append(track_data_obj.values.flatten())
    return np.concatenate(all_values) if all_values else np.array([])

def process_uniprot_id(uniprot_id, uniprot_to_mutations, max_transcripts_to_check, api_key):
    """
    Worker function to process a single UniProt ID.
    Initializes its own API clients to be process-safe.
    """
    # Each worker process must have its own API clients
    u_service = UniProt()
    e_service = Ensembl()
    ag_model_client = dna_client.create(api_key)
    
    local_delta_vectors = {}

    try:
        ensembl_gene_id = get_ensembl_gene_id(u_service, uniprot_id)
        if not ensembl_gene_id: return None

        gene_data = e_service.get_lookup_by_id(ensembl_gene_id, expand=1)
        if not (isinstance(gene_data, dict) and gene_data.get('assembly_name') == 'GRCh38'): return None

        fasta_data = u_service.retrieve(uniprot_id, "fasta")
        if not isinstance(fasta_data, str): return None
        uniprot_seq = "".join(fasta_data.splitlines()[1:])

        transcript, reason = find_best_matching_transcript(e_service, gene_data, uniprot_seq, max_transcripts_to_check)
        if not transcript:
            # tqdm.write is not process-safe, so we return the error to be printed in the main thread
            return f"Warning: Could not find a suitable transcript for {uniprot_id}. Reason: {reason}"

        gene_center = (gene_data['start'] + gene_data['end']) // 2
        pred_start, pred_end = gene_center - 262144, gene_center + 262144
        chromosome = "chr" + gene_data['seq_region_name']
        interval = genome.Interval(chromosome=chromosome, start=pred_start, end=pred_end)

        for mutation in uniprot_to_mutations.get(uniprot_id, []):
            target_id = f"{uniprot_id}_{mutation}"
            variant_info, reason = get_variant_info(uniprot_seq, transcript, mutation, gene_data['strand'])
            
            if not variant_info:
                tqdm.write(f"Warning: Could not map mutation {mutation} to genome for {uniprot_id}. Reason: {reason}")
                continue
            
            variant = genome.Variant(
                chromosome=chromosome, 
                position=variant_info['position'],
                reference_bases=variant_info['reference_bases'],
                alternate_bases=variant_info['alternate_bases']
            )

            variant_predictions = ag_model_client.predict_variant(
                interval=interval, variant=variant,
                requested_outputs=list(dna_client.OutputType),
                ontology_terms=['EFO:0002067']
            )

            if variant_predictions:
                ref_vector = flatten_predictions_to_vector(variant_predictions.reference)
                alt_vector = flatten_predictions_to_vector(variant_predictions.alternate)
                
                if ref_vector.size > 0 and alt_vector.size > 0 and ref_vector.shape == alt_vector.shape:
                    delta_vector = alt_vector - ref_vector
                    local_delta_vectors[target_id] = delta_vector
            time.sleep(0.2) # Add a slightly longer delay to be kind to APIs in parallel
            
    except Exception as e:
        return f"An unexpected error occurred while processing {uniprot_id}: {e} (Line: {e.__traceback__.tb_lineno})"

    return local_delta_vectors


def main(args):
    print("--- Starting AlphaGenome Delta Vector Featurization Pipeline ---")

    df = pd.concat([pd.read_parquet(p) for p in args.input_paths])
    unique_uniprots = df['uniprot_id'].unique()
    uniprot_to_mutations = df.groupby('uniprot_id')['mutation'].unique().to_dict()
    print(f"Found {len(df[['uniprot_id', 'mutation']].drop_duplicates())} unique (uniprot_id, mutation) pairs to process.")

    api_key = os.environ.get('ALPHAGENOME_API_KEY')
    if not api_key: raise ValueError("ALPHAGENOME_API_KEY not set.")

    delta_vectors = {}
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs to the pool
        future_to_uniprot = {executor.submit(process_uniprot_id, uid, uniprot_to_mutations, args.max_transcripts_to_check, api_key): uid for uid in unique_uniprots}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_uniprot), total=len(unique_uniprots), desc="Processing Proteins"):
            result = future.result()
            if isinstance(result, dict):
                delta_vectors.update(result)
            elif isinstance(result, str): # It's an error message
                tqdm.write(result)
            
    print(f"\nSuccessfully generated {len(delta_vectors)} delta vectors.")
    print(f"Saving vectors to '{args.output_path}'...")
    np.savez_compressed(args.output_path, **delta_vectors)
    print("--- Done ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AlphaGenome delta vectors for mutant proteins.")
    parser.add_argument('--input_paths', nargs='+', required=True, help='Path(s) to the Parquet split files.')
    parser.add_argument('--output_path', type=str, default='mutant_delta_vectors.npz')
    parser.add_argument('--max_transcripts_to_check', type=int, default=25,
                        help='Maximum number of longest, protein-coding transcripts to check per gene for alignment.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel worker processes to use.')
    args = parser.parse_args()
    main(args)


