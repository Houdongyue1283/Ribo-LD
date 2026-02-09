def calculate_sequence_identity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Error! Not the same length!")
    
    # Count identical characters
    identical_count = sum(1 for a, b in zip(seq1, seq2) if a == b)
    
    # Compute identity ratio
    identity = identical_count / len(seq1)
    
    return identity
def parse_dot_bracket(dot_bracket):
    """
    Parse a secondary structure dot-bracket string into a set of base-pair indices.
    
    Args:
        dot_bracket (str): Dot-bracket representation of the secondary structure.
    
    Returns:
        set: Set of base pairs in the form {(i, j), ...}.
    """
    stack = []
    pairs = set()
    for i, char in enumerate(dot_bracket):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs.add((j, i))  # Base-pair indices
    return pairs


def calculate_f1_score(true_structure, predicted_structure):
    """
    Compute the F1 score between two dot-bracket secondary structures.
    
    Args:
        true_structure (str): Ground-truth dot-bracket secondary structure.
        predicted_structure (str): Predicted dot-bracket secondary structure.
    
    Returns:
        float: F1 score.
    """
    # Parse paired bases from dot-bracket structures
    true_pairs = parse_dot_bracket(true_structure)
    predicted_pairs = parse_dot_bracket(predicted_structure)
    
    # Compute True Positive, False Positive, False Negative
    tp = len(true_pairs & predicted_pairs)  # Correctly predicted base pairs
    fp = len(predicted_pairs - true_pairs)  # Incorrectly predicted base pairs
    fn = len(true_pairs - predicted_pairs)  # Missed base pairs
    
    # Precision and recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1


def validate_base_pairing(sequence: str, structure: str) -> float:
    """
    Compute the fraction of predicted base pairs that follow complementarity rules.

    Args:
        sequence: RNA sequence.
        structure: Dot-bracket secondary structure representation.

    Returns:
        Fraction of correctly paired bases among all predicted pairs.
    """
    valid_pairs = {"AU", "UA", "GC", "CG", "GU", "UG"}
    stack = []
    correct_pairs = 0
    total_pairs = 0
    
    # Record pairing indices
    pair_map = {}
    for i, char in enumerate(structure):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pair_map[j] = i
                pair_map[i] = j
    
    # Compute fraction of pairs that follow pairing rules
    for i, j in pair_map.items():
        if i < j:  # Avoid double counting
            total_pairs += 1
            if sequence[i] + sequence[j] in valid_pairs:
                correct_pairs += 1
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0