import random

# Base prompt instructing the VLM on its role
SYSTEM_PROMPT = """You are an intelligent visual search assistant. The user wants to search for a new image starting from a reference image.
You will be provided with two images: [Image 1] (the reference) and [Image 2] (the target).
You must output a highly concise, natural, intent-oriented modification text that a human would write to transition from Image 1 to Image 2.
"""

# The 6 MiERDCIR Intents
INTENT_PROMPTS = {
    "a_instance_level": """
Scenario: Instance-level modification.
Focus purely on changing the visual properties of a specific object (e.g., color, texture, style) without mentioning its functionality.
Do NOT describe it like a machine (e.g., 'predominantly black'), just write a short practical command (e.g., "Change the laptop to a black single version").
Keep it brief and conversational.
""",

    "b_global_view": """
Scenario: Global visual or contextual change.
Focus on the overall atmosphere, lighting, background, or vibe of the image rather than specific objects.
Keep details minimal. Output a command like "Make it look like a studio photography shot" or "Change the vibe to more professional".
""",

    "c_functional_affordance": """
Scenario: Functional or Affordance modification.
Focus on making the object suitable for a specific purpose or action, rather than just describing its look.
For example, instead of saying "make the heels higher", say "make these shoes suitable for a formal evening gala".
Output a short, intent-based command.
""",

    "d_negative_constraint": """
Scenario: Negative Constraint (Exclusionary intent).
The user wants to specify what they DO NOT want.
Formulate the instruction using a negation. For example: "Change the dress, but no red colors please" or "Keep the style but definitely avoid leather".
Give a realistic, brief exclusion based on the difference between Image 1 and Image 2.
""",

    "e_comparative_intensity": """
Scenario: Comparative / Relative Intensity.
The user wants a directional change in a continuous attribute (e.g., darker, longer, larger, brighter, more vintage).
Use relative terms. For example: "Make the color a bit deeper" or "Make the skirt slightly longer".
Do not give absolute descriptions.
""",

    "f_spatial_positional": """
Scenario: Spatial & Positional Logic.
Focus on where objects are located or their relative placement.
Formulate an instruction based on layout changes (if any) or imaginary scene moving.
For example: "Move the bag on the left more towards the center" or "Scale down the background trees".
Keep it practical.
"""
}

# The keys as an easy-to-use list
INTENT_SCENARIOS = list(INTENT_PROMPTS.keys())

def get_random_intent_prompt() -> tuple[str, str]:
    """
    Randomly selects one of the 6 MiERDCIR intents.
    Returns:
        tuple: (intent_name, intent_prompt)
    """
    intent_key = random.choice(INTENT_SCENARIOS)
    return intent_key, INTENT_PROMPTS[intent_key]

def build_qwen2vl_prompt(original_modification: str, intent_prompt: str) -> list:
    """
    Build a structured prompt suitable for Qwen3 or Llava architectures.
    We pass both images sequentially, followed by the text instructions.
    
    (Note: For Qwen, multiple images in a chat template are supported natively via
    multiple `<|image|>` placeholder tokens in the user content.)
    """
    user_text = (
        f"You are given two images. The first is the reference image, the second is the desired target image.\n"
        f"The original basic description of the change was: '{original_modification}'.\n"
        f"Your task is to re-write this into a natural human search query according to the following scenario constraints:\n\n"
        f"{intent_prompt}\n\n"
        f"Output ONLY the newly generated text query. Do not include quotes, explanations, or introductory phrases."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": user_text}
        ]}
    ]
    return messages
