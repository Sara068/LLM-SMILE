from utils import *

def cell_algorithm(x0, client, split_k=1, delta=0.2, model="gpt-3.5-turbo"):
    """
    Implements the CELL algorithm from the paper, plus iteration tracking.

    x0: the original prompt
    split_k: number of words per chunk
    delta: threshold for a successful contrast score
    model: ChatGPT model to use
    """
    iteration_log = []  # Track each masking attempt for visualization

    # Split the prompt into substrings.
    substrings = split_prompt(x0, split_k)
    # Create a list of indices that have not been masked yet.
    remaining_indices = list(range(len(substrings)))
    # Start with the original prompt.
    x_current = x0

    # Get the response for the current prompt.
    y_current = get_llm_response(x_current, client, model=model)
    print("Original Prompt:", x_current)
    print("Original Response:", y_current)

    # Log the original (no-mask) iteration
    iteration_log.append({
        "iteration": 0,
        "mask_index": None,
        "perturbed_prompt": x_current,
        "perturbed_response": y_current,
        "score": None
    })

    # Start searching for a contrastive prompt
    iteration_count = 1
    while remaining_indices:
        best_score = -1.0
        best_index = None
        best_x = None
        best_y = None

        # Get the current substrings from the current prompt.
        current_substrings = split_prompt(x_current, split_k)

        for j in remaining_indices:
            # Mask the j-th substring.
            masked_substrings = mask_substring(current_substrings, j)
            prompt_with_mask = join_substrings(masked_substrings)

            # Use infilling to replace the <mask> token.
            xj = infill_prompt(prompt_with_mask, client, model=model)
            # Get the LLM's response to the perturbed prompt.
            yj = get_llm_response(xj, client, model=model)
            # Compute the contrast score.
            score = score_contrast(y_current, yj)

            print(f"\nTrying mask at index {j}:")
            print("Perturbed Prompt:", xj)
            print("Perturbed Response:", yj)
            print("Score:", score)

            # Log each attempt
            iteration_log.append({
                "iteration": iteration_count,
                "mask_index": j,
                "perturbed_prompt": xj,
                "perturbed_response": yj,
                "score": score
            })
            iteration_count += 1

            if score > best_score:
                best_score = score
                best_index = j
                best_x = xj
                best_y = yj

        # If the best score exceeds our threshold delta, we return the contrastive explanation + logs.
        if best_score >= delta:
            print("\nContrastive explanation found!")
            return {
                "original_prompt": x0,
                "original_response": y_current,
                "contrastive_prompt": best_x,
                "contrastive_response": best_y,
                "contrast_score": best_score,
                "iterations": iteration_log
            }
        else:
            # If no sufficient contrast was found, update the prompt and remove the used index.
            print(f"\nNo sufficient contrast found in this iteration. Best score: {best_score}")
            x_current = best_x
            y_current = best_y
            remaining_indices.remove(best_index)

    print("\nNO SOLUTION FOUND")
    return None

