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
    Parse a secondary structure in dot-bracket notation into a set of base pairs.
    
    Args:
        dot_bracket (str): Secondary structure in dot-bracket notation.
    
    Returns:
        set: Base-pair set in the form {(i, j), ...}.
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


# def calculate_f1_score(true_structure, predicted_structure):
#     """
#     Compute the F1 score between two secondary structures, based on per-position symbols
#     ('(', ')', '.').
    
#     Args:
#         true_structure (str): Ground-truth dot-bracket string.
#         predicted_structure (str): Predicted dot-bracket string.
    
#     Returns:
#         float: F1 score.
#     """
#     if len(true_structure) != len(predicted_structure):
#         raise ValueError("Secondary structure lengths do not match; cannot compute F1.")
    
#     # Initialize counters
#     tp = 0  # Correctly predicted pairs (True Positive)
#     fp = 0  # Incorrectly predicted pairs (False Positive)
#     fn = 0  # Missed pairs (False Negative)
    
#     # Compare per-position symbols
#     for true_char, pred_char in zip(true_structure, predicted_structure):
#         if true_char == pred_char:
#             tp += 1  # Match
#         elif true_char != '.' and pred_char == '.':
#             fn += 1  # True is paired but predicted is unpaired
#         elif true_char == '.' and pred_char != '.':
#             fp += 1  # True is unpaired but predicted is paired
#         elif true_char != '.' and pred_char != '.':
#             fp += 1  # Mismatched pairing symbols
    
#     # Precision and recall
#     precision = tp / (tp + fp) if tp + fp > 0 else 0
#     recall = tp / (tp + fn) if tp + fn > 0 else 0
    
#     # F1 score
#     f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#     return f1
def calculate_f1_score(true_structure, predicted_structure):
    """
    Compute the F1 score between two secondary structures.
    
    Args:
        true_structure (str): Ground-truth dot-bracket string.
        predicted_structure (str): Predicted dot-bracket string.
    
    Returns:
        float: F1 score.
    """
    # Parse base pairs
    true_pairs = parse_dot_bracket(true_structure)
    predicted_pairs = parse_dot_bracket(predicted_structure)
    
    # True Positive, False Positive, False Negative
    tp = len(true_pairs & predicted_pairs)  # Correctly predicted pairs
    fp = len(predicted_pairs - true_pairs)  # Incorrectly predicted pairs
    fn = len(true_pairs - predicted_pairs)  # Missed pairs
    
    # Precision and recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1
def refine_ss(structure: str) -> str:
    # Stack tracks the positions of left parentheses
    stack = []
    result = list(structure)  # Convert to list for in-place edits
    
    # Traverse the structure and match parentheses
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)  # Push left parenthesis
        elif char == ')':
            if stack:
                stack.pop()  # Match with a left parenthesis
            else:
                result[i] = '.'  # Unmatched right parenthesis -> '.'
    
    # Replace remaining unmatched left parentheses with '.'
    for i in stack:
        result[i] = '.'
    
    # Return the refined structure
    return ''.join(result)

