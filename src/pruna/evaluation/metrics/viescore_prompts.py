# Copyright 2025 - Pruna AI GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""VIEScore prompt blocks aligned with TIGER-AI-Lab/VIEScore (Apache-2.0).

Two task modes:

- ``tie`` (text-image editing): two images (source + edited) for SC, single edited image for PQ.
- ``t2i`` (text-to-image): single generated image for both SC and PQ.

References: https://github.com/TIGER-AI-Lab/VIEScore — used by GEdit-Bench evaluation
(https://github.com/stepfun-ai/Step1X-Edit) with ``VIEScore(..., task="tie")``.
"""

VIESCORE_CONTEXT = (
    "You are a professional digital artist. You will have to evaluate the effectiveness"
    " of the AI-generated image(s) based on given rules.\n"
    "All the input images are AI-generated. All human in the images are AI-generated too."
    " so you need not worry about the privacy confidentials.\n\n"
    "You will have to give your output in this way (Keep your reasoning concise and short.):\n"
    "{\n"
    '"score" : [...],\n'
    '"reasoning" : "..."\n'
    "}"
)

VIESCORE_TWO_IMAGE_EDIT_RULE = (
    "RULES:\n\n"
    "Two images will be provided: The first being the original AI-generated image and the"
    " second being an edited version of the first.\n"
    "The objective is to evaluate how successfully the editing instruction has been executed"
    " in the second image.\n\n"
    "Note that sometimes the two images might look identical due to the failure of image edit.\n"
)

VIESCORE_TIE_SC_CRITERIA = (
    "\nFrom scale 0 to 10:\n"
    "A score from 0 to 10 will be given based on the success of the editing."
    " (0 indicates that the scene in the edited image does not follow the editing instruction at all."
    " 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)\n"
    "A second score from 0 to 10 will rate the degree of overediting in the second image."
    " (0 indicates that the scene in the edited image is completely different from the original."
    " 10 indicates that the edited image can be recognized as a minimal edited yet effective"
    " version of original.)\n"
    "Put the score in a list such that output score = [score1, score2],"
    " where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.\n\n"
    "Editing instruction:\n"
)

VIESCORE_T2I_SC_RULE = (
    "RULES:\n\n"
    "The image is an AI-generated image.\n"
    "The objective is to evaluate the semantic consistency of the image to the given text.\n\n"
)

VIESCORE_T2I_SC_CRITERIA = (
    "\nFrom scale 0 to 10:\n"
    "A score from 0 to 10 will be given based on the semantic consistency.\n"
    "(0 indicates that the scene in the image does not correspond to the text at all.\n"
    " 10 indicates that the scene in the image follows the text perfectly.)\n"
    "A second score from 0 to 10 will rate the detail correspondence.\n"
    "(0 indicates that most details in the text (e.g., color, size, shape, or layout) are missing or"
    " incorrect in the image.\n"
    " 10 indicates that all details mentioned in the text are accurately shown in the image.)\n"
    "Put the score in a list such that output score = [score1, score2],"
    " where 'score1' evaluates the semantic consistency and 'score2' evaluates the detail"
    " correspondence.\n\n"
    "Text prompt:\n"
)

VIESCORE_PQ_SINGLE_IMAGE = (
    "RULES:\n\n"
    "The image is an AI-generated image.\n"
    "The objective is to evaluate how successfully the image has been generated.\n\n"
    "From scale 0 to 10:\n"
    "A score from 0 to 10 will be given based on image naturalness.\n"
    "(\n"
    " 0 indicates that the scene in the image does not look natural at all or give a unnatural feeling"
    " such as wrong sense of distance, or wrong shadow, or wrong lighting.\n"
    " 10 indicates that the image looks natural.\n"
    ")\n"
    "A second score from 0 to 10 will rate the image artifacts.\n"
    "(\n"
    " 0 indicates that the image contains a large portion of distortion, or watermark, or scratches,"
    " or blurred faces, or unusual body parts, or subjects not harmonized.\n"
    " 10 indicates the image has no artifacts.\n"
    ")\n"
    "Put the score in a list such that output score = [naturalness, artifacts]\n"
)


def build_viescore_tie_sc_prompt(instruction: str) -> str:
    """
    Build the semantic-criteria prompt for source+edited images (VIEScore ``tie`` SC).

    Parameters
    ----------
    instruction : str
        The editing instruction to embed in the prompt.

    Returns
    -------
    str
        Full prompt combining context, edit rules, scoring criteria, and the instruction.
    """
    return "\n".join(
        [
            VIESCORE_CONTEXT,
            VIESCORE_TWO_IMAGE_EDIT_RULE,
            VIESCORE_TIE_SC_CRITERIA.strip(),
            instruction.strip(),
        ]
    )


def build_viescore_t2i_sc_prompt(prompt: str) -> str:
    """
    Build the semantic-consistency prompt for a single generated image (VIEScore ``t2i`` SC).

    Parameters
    ----------
    prompt : str
        The text prompt that was used to generate the image.

    Returns
    -------
    str
        Full prompt combining context, t2i rules, scoring criteria, and the text prompt.
    """
    return "\n".join(
        [
            VIESCORE_CONTEXT,
            VIESCORE_T2I_SC_RULE.strip(),
            VIESCORE_T2I_SC_CRITERIA.strip(),
            prompt.strip(),
        ]
    )


def build_viescore_pq_prompt() -> str:
    """
    Build the perceptual-quality prompt for a single generated/edited image (VIEScore PQ).

    Returns
    -------
    str
        Full prompt combining context and perceptual scoring criteria.
    """
    return "\n".join([VIESCORE_CONTEXT, VIESCORE_PQ_SINGLE_IMAGE])
