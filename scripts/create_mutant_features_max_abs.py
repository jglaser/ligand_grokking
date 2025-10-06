"""
This script implements the core "delta vector" featurization pipeline using the
efficient `predict_variant` method from the AlphaGenome API.

Workflow:
1.  Identifies all unique (uniprot_id, mutation) pairs from the input data.
2.  **Processes each unique UniProt ID in parallel to speed up execution.**
3.  For each protein, it finds the best-matching Ensembl transcript by aligning
    against the UniProt sequence.
4.  For each mutation, it uses this best-match transcript to determine the precise
    genomic location and the required nucleotide change(s). It now correctly
    handles both single (SNV) and multi-nucleotide (MNV) variants.
5.  It makes a single, efficient call to `ag_model_client.predict_variant()`.
6.  It creates the final "delta vector" (mutant - wild_type).
"""
import argparse
import pandas as pd
import numpy as np
from scipy import sparse
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
    Iterates through transcripts and returns the one with the best alignment score
    using a robust "glocal" alignment strategy.
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

    # --- FINAL ALIGNMENT STRATEGY: Use a "glocal" alignment ---
    aligner = PairwiseAligner()
    # Use global alignment to consider the entire sequence
    aligner.mode = 'global'
    # Do not penalize gaps at the ends of the sequences. This finds the best
    # alignment of the UniProt sequence *within* the Ensembl transcript.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    # Standard scoring for match/mismatch/gaps
    aligner.match_score = 5
    aligner.mismatch_score = -4
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    # --- END ---

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
            # Clean the translated sequence of any leading/trailing unknown codons
            ensembl_protein_seq = "".join([CODON_TO_AA.get(sequence[i:i+3], '?') for i in range(0, len(sequence), 3)]).strip('?')

            try:
                # Find the best score for this transcript
                score = aligner.score(ensembl_protein_seq, uniprot_seq)
                if score > best_score:
                    best_score = score
                    best_transcript = full_transcript_data
            except Exception:
                continue

    # Heuristic: score should be at least 80% of a perfect match to be considered valid
    if best_score < 0.8 * len(uniprot_seq) * 5:
        return None, f"No transcript found with a high-quality alignment (best score: {best_score})."

    return best_transcript, None

def reverse_complement(dna_seq):
    """Computes the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return "".join(complement.get(base, base) for base in reversed(dna_seq))

def get_variant_info(uniprot_protein_seq, transcript_data, mutation, gene_strand):
    """
    Translates a protein mutation to a genomic variant using a hybrid,
    context-first strategy to ensure alignment correctness.
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

        # --- HYBRID ALIGNMENT AND VERIFICATION ---
        aligner = PairwiseAligner()
        aligner.mode = 'global'
        aligner.target_end_gap_score = 0.0
        aligner.query_end_gap_score = 0.0
        aligner.match_score = 5
        aligner.mismatch_score = -4
        aligner.open_gap_score = -10
        aligner.extend_gap_score = -1

        corrected_pos_idx = -1

        # 1. First Pass: Use global alignment
        try:
            first_alignment = next(aligner.align(ensembl_protein_seq, uniprot_protein_seq))
            ensembl_start = first_alignment.coordinates[0, 0]
            uniprot_start = first_alignment.coordinates[1, 0]
            offset = ensembl_start - uniprot_start
            corrected_pos_idx = pos_idx + offset
        except StopIteration:
            return None, "Primary alignment failed."

        # 2. Strong Verification: Check local context
        window = 5
        uniprot_context_start = max(0, pos_idx - window)
        uniprot_context_end = min(len(uniprot_protein_seq), pos_idx + window + 1)
        uniprot_context = uniprot_protein_seq[uniprot_context_start:uniprot_context_end]

        ensembl_context_start = max(0, corrected_pos_idx - window)
        ensembl_context_end = min(len(ensembl_protein_seq), corrected_pos_idx + window + 1)
        ensembl_context = ensembl_protein_seq[ensembl_context_start:ensembl_context_end]

        # 3. Fallback: If context mismatches, perform a direct search
        if uniprot_context != ensembl_context:
            try:
                # Search for the exact context from UniProt within the Ensembl sequence
                fallback_start_idx = ensembl_protein_seq.index(uniprot_context)
                # If found, recalculate the corrected position based on this manual match
                corrected_pos_idx = fallback_start_idx + (pos_idx - uniprot_context_start)
            except ValueError:
                # If the context is not found anywhere, the transcript is a bad match
                tqdm.write(f"!! CRITICAL WARNING: Context mismatch for {mutation} in {transcript_data['id']}. Align and search failed.")
                return None, "Alignment context mismatch and fallback search failed."
        # --- END OF HYBRID STRATEGY ---

        corrected_pos = corrected_pos_idx + 1

        if not (0 <= corrected_pos_idx < len(ensembl_protein_seq)):
            return None, f"Position {pos} (corrected to {corrected_pos}) is out of bounds for Ensembl protein."

        found_aa = ensembl_protein_seq[corrected_pos_idx]
        if found_aa != ref_aa:
            return None, f"Reference AA mismatch at aligned position {corrected_pos}. Expected {ref_aa}, but transcript has {found_aa}."

        codon_start_in_cds = (corrected_pos - 1) * 3
        original_codon = cds[codon_start_in_cds : codon_start_in_cds + 3]

        new_codon = CODON_TABLE.get(alt_aa, [None])[0]
        if new_codon is None: return None, f"Invalid alternate amino acid '{alt_aa}'."
        if original_codon == new_codon: return None, "Mutation is synonymous (no nucleotide change)."

        diffs = [i for i, (c1, c2) in enumerate(zip(original_codon, new_codon)) if c1 != c2]

        if len(diffs) == 1:
            diff_idx = diffs[0]
            ref_bases = original_codon[diff_idx]
            alt_bases = new_codon[diff_idx]
            cds_pos_of_mutation = codon_start_in_cds + diff_idx
        else:
            ref_bases = original_codon
            alt_bases = new_codon
            cds_pos_of_mutation = codon_start_in_cds

        cds_exons = sorted(transcript_data['Exon'], key=lambda x: x['start'])
        genomic_pos_of_mutation = -1

        if gene_strand == 1: # Positive strand
            bases_covered = 0
            for exon in cds_exons:
                exon_len = exon['end'] - exon['start'] + 1
                if bases_covered + exon_len > cds_pos_of_mutation:
                    offset_in_exon = cds_pos_of_mutation - bases_covered
                    genomic_pos_of_mutation = exon['start'] + offset_in_exon
                    break
                bases_covered += exon_len
        else: # Negative strand
            bases_covered = 0
            for exon in sorted(cds_exons, key=lambda x: x['start'], reverse=True):
                exon_len = exon['end'] - exon['start'] + 1
                if bases_covered + exon_len > cds_pos_of_mutation:
                    offset_in_exon = cds_pos_of_mutation - bases_covered
                    genomic_pos_of_mutation = exon['end'] - offset_in_exon
                    break
                bases_covered += exon_len

            # **FIX**: Reverse-complement the bases for the negative strand
            ref_bases = reverse_complement(ref_bases)
            alt_bases = reverse_complement(alt_bases)


        if genomic_pos_of_mutation == -1:
            return None, "Failed to map CDS position to a genomic coordinate."

        result = {
            "position": genomic_pos_of_mutation,
            "reference_bases": ref_bases,
            "alternate_bases": alt_bases
        }
        return result, None
    except Exception as e:
        return None, f"An unexpected exception occurred during variant mapping: {e} (Line: {e.__traceback__.tb_lineno})"

# Corrected dictionary for track resolutions in base pairs
TRACK_RESOLUTIONS = {
    'atac': 1,
    'cage': 1,
    'chip_histone': 128,  # 524288 / 4096 = 128
    'chip_tf': 128,       # 524288 / 4096 = 128
    'contact_maps': 2048, # Resolution doesn't matter as it's empty
    'dnase': 1,
    'procap': 1,
    'rna_seq': 1,
    'splice_site_usage': 1,
    'splice_sites': 1
}

# Corrected dictionary for feature slices to ensure a constant 335-dim vector
# Arranged alphabetically to match the processing loop
TRACK_FEATURE_SLICES = {
    'atac': slice(0, 1),
    'cage': slice(1, 3),
    'chip_histone': slice(3, 13),
    'chip_tf': slice(13, 319),
    'dnase': slice(319, 320),
    'procap': slice(320, 322),
    'rna_seq': slice(322, 327),
    'splice_site_usage': slice(327, 331),
    'splice_sites': slice(331, 335)
}
# Total features: 335

# --- REPLACE THE OLD FUNCTION WITH THIS NEW VERSION ---

def max_abs_diff_pooling_with_mask(predictions: dna_output.Output,
                                   prediction_interval: genome.Interval,
                                   transcript_data: dict) -> np.ndarray:
    """
    Applies a resolution-aware spatial mask and scatters track features into a
    fixed-size vector, ensuring a constant output shape.
    """
    # 1. Create a fixed-size zero vector for the final output
    output_vector = np.zeros(335, dtype=np.float32)

    # Define the mask boundaries from the transcript data (1-bp resolution)
    mask_start_genomic = transcript_data['start']
    mask_end_genomic = transcript_data['end']
    pred_window_start = prediction_interval.start

    base_start_index = max(0, mask_start_genomic - pred_window_start)
    base_end_index = min(prediction_interval.end - pred_window_start, mask_end_genomic - pred_window_start)

    if base_start_index >= base_end_index:
        return output_vector # Return the zero vector if no overlap

    # Iterate through all available prediction tracks
    for modality in sorted(TRACK_FEATURE_SLICES.keys()):
        if not hasattr(predictions.reference, modality):
            continue

        ref_track_obj = getattr(predictions.reference, modality)
        alt_track_obj = getattr(predictions.alternate, modality)

        if (hasattr(ref_track_obj, 'values') and ref_track_obj.values is not None and
            hasattr(alt_track_obj, 'values') and alt_track_obj.values is not None):

            # 2. Scale mask indices for the current track's resolution
            resolution = TRACK_RESOLUTIONS.get(modality, 1)
            start_index = base_start_index // resolution
            end_index = base_end_index // resolution

            if start_index >= end_index:
                continue # Skip if no overlap at this resolution

            masked_ref = ref_track_obj.values[start_index:end_index]
            masked_alt = alt_track_obj.values[start_index:end_index]

            if masked_ref.shape != masked_alt.shape or masked_ref.size == 0:
                continue

            abs_diff = np.abs(masked_alt - masked_ref)
            pooled_delta = np.max(abs_diff, axis=0)

            if pooled_delta.ndim > 1:
                pooled_delta = pooled_delta.flatten()
            
            # 3. Scatter the calculated features into the correct slice
            feature_slice = TRACK_FEATURE_SLICES[modality]
            if pooled_delta.size == (feature_slice.stop - feature_slice.start):
                 output_vector[feature_slice] = pooled_delta

    return output_vector

def inspect_feature_shapes(predictions: dna_output.Output):
    """
    Iterates through prediction tracks and prints their name, shape, and feature width.
    """
    print("\n--- Inspecting Feature Shapes ---")
    total_features = 0
    for modality in sorted([m for m in dir(predictions.reference) if isinstance(getattr(predictions.reference, m), track_data.TrackData)]):
        track_obj = getattr(predictions.reference, modality)
        if hasattr(track_obj, 'values') and track_obj.values is not None:
            shape = track_obj.values.shape

            # Calculate width based on the number of dimensions
            if len(shape) == 2:
                width = shape[1]
            elif len(shape) == 3:
                width = shape[1] * shape[2]
            else:
                width = 0 # Or handle other cases as needed

            total_features += width
            print(f"- Track: {modality:<30} Shape: {str(shape):<20} Feature Width: {width}")

    print(f"--- Total Features: {total_features} ---\n")

def process_uniprot_id(uniprot_id, uniprot_to_mutations, max_transcripts_to_check, feature_type, api_key, random_seed):
    """
    Worker function to process a single UniProt ID.
    Initializes its own API clients to be process-safe.
    """
    u_service = UniProt()
    e_service = Ensembl()
    ag_model_client = dna_client.create(api_key)

    local_feature_vectors = {}

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
            return f"Warning: Could not find a suitable transcript for {uniprot_id}. Reason: {reason}"

        gene_center = (gene_data['start'] + gene_data['end']) // 2
        pred_start, pred_end = gene_center - 262144, gene_center + 262144
        chromosome = "chr" + gene_data['seq_region_name']
        interval = genome.Interval(chromosome=chromosome, start=pred_start, end=pred_end)

        mutations = uniprot_to_mutations.get(uniprot_id, [])

        if feature_type == 'randomized_delta':
            rng = np.random.default_rng(random_seed + pred_start)
            shuffled_mutations = rng.choice(mutations, size=len(mutations), replace=False)
        else:
            shuffled_mutations = mutations

        for mutation, mutation_label  in zip(mutations, shuffled_mutations):
            target_id = f"{uniprot_id}_{mutation_label}"
            all_mutations = mutation.split(',')

            for point_mutation in all_mutations:
                variant_info, reason = get_variant_info(uniprot_seq, transcript, point_mutation, gene_data['strand'])
                if not variant_info:
                    tqdm.write(f"Warning: Could not map mutation {point_mutation} to genome for {uniprot_id}. Reason: {reason}")
                    continue

                variant = genome.Variant(
                    chromosome=chromosome,
                    position=variant_info['position'],
                    reference_bases=variant_info['reference_bases'],
                    alternate_bases=variant_info['alternate_bases']
                )

                requested_outputs = [dna_client.OutputType[track.upper()] for track in args.use_tracks] if args.use_tracks else list(dna_client.OutputType)

                variant_predictions = ag_model_client.predict_variant(
                    interval=interval, variant=variant,
                    requested_outputs=requested_outputs,
                    ontology_terms=['EFO:0002067']
                )

                if variant_predictions:
                    # uncomment to see feature shapes
                    # inspect_feature_shapes(variant_predictions)

                    out_vec = None
                    # --- FEATURE PROCESSING LOGIC ---
                    if feature_type == 'masked_max_abs_diff':
                        out_vec = max_abs_diff_pooling_with_mask(variant_predictions, interval, transcript)
                    elif feature_type in ['delta', 'mutant_only', 'wt_only', 'randomized_delta']:
                        # Original logic with mean reduction over the whole window
                        ref_vector = mean_reduce_predictions_to_vector(variant_predictions.reference)
                        alt_vector = mean_reduce_predictions_to_vector(variant_predictions.alternate)

                        if feature_type == 'delta' or feature_type == 'randomized_delta':
                            if ref_vector.size > 0 and alt_vector.size > 0 and ref_vector.shape == alt_vector.shape:
                                out_vec = alt_vector - ref_vector
                        elif feature_type == 'mutant_only':
                            if alt_vector.size > 0:
                                out_vec = alt_vector
                        elif feature_type == 'wt_only':
                            if ref_vector.size > 0:
                                out_vec = ref_vector
                    else:
                        raise ValueError(f"Invalid --feature_type {feature_type}")

                    if out_vec is not None and out_vec.size > 0:
                        # approximate the effect of a compound mutatation assuming additivity
                        if target_id in local_feature_vectors:
                            local_feature_vectors[target_id] += out_vec
                        else:
                            local_feature_vectors[target_id] = out_vec

            time.sleep(0.2)

    except Exception as e:
        return f"An unexpected error occurred while processing {uniprot_id}: {e} (Line: {e.__traceback__.tb_lineno})"

    return local_feature_vectors


def main(args):
    print("--- Starting AlphaGenome Delta Vector Featurization Pipeline ---")

    df = pd.concat([pd.read_parquet(p) for p in args.input_paths])
    unique_uniprots = df['uniprot_id'].unique()
    uniprot_to_mutations = df.groupby('uniprot_id')['mutation'].unique().to_dict()
    print(f"Found {len(df[['uniprot_id', 'mutation']].drop_duplicates())} unique (uniprot_id, mutation) pairs to process.")

    api_key = os.environ.get('ALPHAGENOME_API_KEY')
    if not api_key: raise ValueError("ALPHAGENOME_API_KEY not set.")

    feature_vectors = {}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_uniprot = {executor.submit(process_uniprot_id, uid, uniprot_to_mutations, args.max_transcripts_to_check,
                                             args.feature_type, api_key, args.random_seed): uid for uid in unique_uniprots}

        for future in tqdm(as_completed(future_to_uniprot), total=len(unique_uniprots), desc="Processing Proteins"):
            result = future.result()
            if isinstance(result, dict):
                feature_vectors.update(result)
            elif isinstance(result, str):
                tqdm.write(result)

    print(f"\nSuccessfully generated {len(feature_vectors)} feature vectors.")

    print(f"Saving vectors to '{args.output_path}'...")
    np.savez_compressed(args.output_path, **feature_vectors)
    print("--- Done ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate AlphaGenome delta vectors for mutant proteins.")
    parser.add_argument('--input_paths', nargs='+', required=True, help='Path(s) to the Parquet split files.')
    parser.add_argument('--output_path', type=str, default='mutant_delta_vectors.npz')
    parser.add_argument('--feature_type', type=str,
                        choices=['delta', 'mutant_only', 'wt_only', 'randomized_delta', 'masked_max_abs_diff'],
                        default='delta', help="""
                        'delta': (mutant - wt) using mean-pooling over the full 524kb window.
                        'masked_max_abs_diff': max(|mutant - wt|) using a spatial mask for the transcript, with zero-padding.
                        """)
    parser.add_argument('--use_tracks', type=str, nargs='+', default=[], help='Feature tracks, e.g. SPLICE_SITES, SPLICE_SITE_USAGE')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--max_transcripts_to_check', type=int, default=25,
                        help='Maximum number of longest, protein-coding transcripts to check per gene for alignment.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel worker processes to use.')
    args = parser.parse_args()
    main(args)
