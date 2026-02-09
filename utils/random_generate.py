import random

def generate_random_rna_sequences(num_samples: int, sequence_length: int):
    """
    Generate a fixed number of random RNA sequences.
    
    Args:
        num_samples (int): Number of RNA sequences to generate.
        sequence_length (int): Length of each RNA sequence.
        
    Returns:
        list: List of randomly generated RNA sequences.
    """
    nucleotides = ['A', 'T', 'G', 'C']
    return [
        ''.join(random.choice(nucleotides) for _ in range(sequence_length))
        for _ in range(num_samples)
    ]

def write_sequences_to_file(sequences, file_path):
    """
    Write a list of RNA sequences to a text file.
    
    Args:
        sequences (list): List of RNA sequences.
        file_path (str): Path to the output file.
    """
    with open(file_path, 'w') as f:
        for seq in sequences:
            f.write(seq + '\n')

# Example usage
num_samples = 10000  # Total number of sequences
sequence_length = 46  # Length of each RNA sequence
output_file = 'random_rna_sequences.txt'  # Output file name

# Generate RNA sequences
rna_sequences = generate_random_rna_sequences(num_samples, sequence_length)

# Write to file
write_sequences_to_file(rna_sequences, output_file)
print(f"{num_samples} RNA sequences written to {output_file}.")
