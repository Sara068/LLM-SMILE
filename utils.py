import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import transforms
import numpy as np

import difflib

def get_llm_response(prompt, client, model="gpt-3.5-turbo", temperature=0.7):
    """
    Call ChatGPT API with the given prompt and return the text response.
    """
    chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="gpt-4o",
     )
    return chat_completion.choices[0].message.content

def infill_prompt(prompt_with_mask, client, model="gpt-3.5-turbo", temperature=0.7):
    """
    Given a prompt with a <mask> token, ask ChatGPT to fill in the missing part.
    """
    instruction = (
        "Please fill in the <mask> token in the following text so that it becomes "
        "natural and fluent:\n\n"
        f"{prompt_with_mask}"
    )
    return get_llm_response(instruction, client, model=model, temperature=temperature)

def score_contrast(y_current, y_perturbed):
    """
    Compute a simple contrast score between two responses.
    Here we use 1 - similarity so that a larger difference yields a higher score.
    """
    similarity = difflib.SequenceMatcher(None, y_current, y_perturbed).ratio()
    return 1 - similarity

def split_prompt(prompt, split_k=1):
    """
    Split the prompt into chunks of split_k words.
    """
    words = prompt.split()
    return [' '.join(words[i:i+split_k]) for i in range(0, len(words), split_k)]

def join_substrings(substrings):
    """
    Reconstruct the prompt from its substrings.
    """
    return " ".join(substrings)

def mask_substring(substrings, index):
    """
    Replace the substring at the given index with a <mask> token.
    Returns a new list of substrings.
    """
    masked = substrings.copy()
    masked[index] = "<mask>"
    return masked

def generate_diff_html(original_text, new_text):
    """
    Produce an HTML-based side-by-side diff of original_text vs. new_text.
    """
    original_lines = original_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.HtmlDiff(wrapcolumn=80).make_table(
        fromlines=original_lines,
        tolines=new_lines,
        fromdesc='Original',
        todesc='Contrastive'
    )
    return diff

# Function to plot text heatmap
def plot_text_heatmap(words, scores, title="", width=10, height=0.4, verbose=0, max_word_per_line=20, word_spacing=20, score_fontsize=10, save_path=None):
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    ax.set_title(title, loc='left')
    cmap = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
    cmap.set_clim(0, 1)
    canvas = ax.figure.canvas
    t = ax.transData
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    loc_y = -0.2
    for i, (token, score) in enumerate(zip(words, scores)):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        text = ax.text(0.0, loc_y, token, bbox={'facecolor': color, 'pad': 5.0, 'linewidth': 1, 'boxstyle': 'round,pad=0.5'}, transform=t, fontsize=14)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        score_text = ax.text(0.01, loc_y - 1, f"{score:.2f}", transform=t, fontsize=score_fontsize, ha='center')
        score_text.draw(canvas.get_renderer())
        ex_score = score_text.get_window_extent()
        if (i+1) % max_word_per_line == 0:
            loc_y -= 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width + word_spacing, units='dots')
    if verbose == 0:
        ax.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_text_heatmap_with_colorbar(
    words,
    scores,
    title="",
    width=10,
    height=0.4,
    vmin=0.7,           # <-- specify min val
    vmax=1.0,           # <-- specify max val
    colormap=plt.cm.bwr,  # any matplotlib colormap
    max_word_per_line=20,
    word_spacing=20,
    score_fontsize=10,
    verbose=0,
    save_path=None
):
    """
    A function that plots text tokens and colors them based on 'scores' using
    a specified colormap and normalization in [vmin, vmax]. Also shows a
    horizontal colorbar below the tokens.
    """

    # 1) Initialize figure and axis
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    ax.set_title(title, loc='left', fontsize=12)

    # 2) Create a Normalize object for [vmin, vmax]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 3) Create a ScalarMappable so we can draw a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # dummy array for colorbar

    canvas = ax.figure.canvas
    t = ax.transData

    loc_y = -0.2  # initial y offset

    for i, (token, score) in enumerate(zip(words, scores)):
        # 4) Convert 'score' to RGBA using colormap & normalization
        rgba = colormap(norm(score))
        # Convert RGBA to a HEX color (#RRGGBB)
        r = int(rgba[0]*255)
        g = int(rgba[1]*255)
        b = int(rgba[2]*255)
        color_hex = f'#{r:02x}{g:02x}{b:02x}'

        # Draw token with colored bbox
        text = ax.text(
            0.0, loc_y, token,
            bbox={
                'facecolor': color_hex,
                'pad': 5.0,
                'linewidth': 1,
                'boxstyle': 'round,pad=0.5'
            },
            transform=t,
            fontsize=14
        )
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()

        # Draw numeric score below each token
        score_text = ax.text(
            0.01, loc_y - 1,
            f"{score:.2f}",
            transform=t,
            fontsize=score_fontsize,
            ha='center'
        )
        score_text.draw(canvas.get_renderer())

        # Move to a new line or shift horizontally
        if (i+1) % max_word_per_line == 0:
            loc_y -= 2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width + word_spacing, units='dots')

    # Hide axis lines/labels if verbose=0
    if verbose == 0:
        ax.axis('off')

    # 5) Add a horizontal colorbar
    #    orientation='horizontal' draws it below or above the axis
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.2, fraction=0.05)
    cbar.set_label("Score", fontsize=12)

    # 6) Save figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()