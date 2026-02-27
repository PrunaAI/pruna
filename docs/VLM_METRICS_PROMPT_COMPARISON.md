# VLM Metrics: Prompt Comparison (Pruna vs InferBench)

Overview of prompt differences between Pruna's VLM metrics and InferBench's implementation.

---

## Summary Table

| Metric | Pruna | InferBench | Key Differences |
|--------|-------|------------|-----------------|
| **Alignment Score** | Single generic question | Multi-question with dependencies | Pruna: 1 prompt; InferBench: N questions from OneIG JSON |
| **VQA** | Same as Alignment (reused) | Dedicated template | Both use "Does this show X? Yes/No" |
| **Text Score** | Short OCR prompt | Detailed OCR prompt | InferBench: longer, explicit format rules |
| **Img Edit Score** | Simple 0–10 rating | Full judge prompts from ImgEdit repo | InferBench: 5-point multi-criteria per edit type |
| **VieScore** | Two short prompts | Long SC + PQ prompts | InferBench: detailed rules, JSON output |
| **QA Accuracy** | Generic "What is in this image?" | Benchmark-specific questions | Different use cases |
| **VLM Base (score)** | Litellm: "Answer Yes or No" / Transformers: "Question: X Answer:" | Generation + logprobs fallback | Response format differs |

---

## 1. Alignment Score

### Pruna
- **Question**: `Does this image show "{prompt}"? Answer Yes or No.`
- **Expected answer**: `Yes`
- **Scope**: Single prompt–image alignment per sample
- **Source**: `metric_alignment_score.py`, `metric_vqa.py` (same logic)

### InferBench
- **Questions**: From OneIG JSON (e.g. `anime.json`, `human.json`, `object.json`)
- **Template**: `{question}. Only answer 'Yes' or 'No'. Do not answer anything else.`
- **Examples**: "Are there boys?", "Are there four boys?", "Is there a nun?", etc.
- **Dependencies**: Parent–child question graph; child scores set to 0 if parent is No
- **Scope**: 9–20 questions per image, dependency-aware aggregation
- **Source**: `alignment_score.py`, `oneig.py` (benchmark)

---

## 2. VQA (Visual Question Answering)

### Pruna
- Same as Alignment Score: `Does this image show "{prompt}"? Answer Yes or No.`
- Used for both `alignment_score` and `vqa` metrics

### InferBench
- **Template**: `Does this figure show "{prompt}"? Please answer yes or no.`
- **Expected answer**: `Yes`
- **Difference**: "figure" vs "image"; "Please answer yes or no" vs "Answer Yes or No"
- **Source**: `vqa.py`

---

## 3. Text Score (OCR)

### Pruna
- **Prompt**: `Extract all text from this image. If no text, say 'No text'.`
- **Output use**: Binary check (no text → score 10.0, else 0.0) — *Note: Pruna text_score appears to use edit distance logic elsewhere; this prompt is for OCR extraction*
- **Source**: `metric_text_score.py`

### InferBench
- **Prompt**:
  ```
  Extract all text visible in this image. Include logos, stylized fonts, handwritten text, and non-standard typography.
  Return only the extracted text, exactly as it appears—no preamble, explanation, or markdown.
  Preserve words, numbers, punctuation, and spacing. If no text is recognized, reply with exactly: No text recognized
  ```
- **Post-processing**: Hallucination removal ("addCriterion", "No text recognized"), Levenshtein vs ground truth, word accuracy
- **Source**: `text_score.py`

---

## 4. Image Edit Score

### Pruna
- **Question**: `Rate 0-10: Does this image show "{prompt}"? Reply with a number.`
- **Input**: Single edited image + prompt
- **Output**: 0–10 score, normalized to [0, 1]
- **Source**: `metric_img_edit_score.py`

### InferBench
- **Input**: Original image + edited image + edit instruction
- **Judge prompts**: Fetched from ImgEdit repo (`prompts.json`) per edit type (replace, add, remove, adjust, style, extract, background, compose)
- **Format**: Long multi-criteria prompts (5-point scale):
  - Prompt Compliance (1–5)
  - Visual Naturalness / Seamlessness (1–5)
  - Physical & Detail Integrity (1–5)
- **Output**: Average of 3 scores, parsed from `"Prompt Compliance: N\nVisual Naturalness: N\n..."` format
- **Source**: `img_edit_score.py`, `img_edit.py` (benchmark), external `prompts.json`

---

## 5. VieScore

### Pruna
- **Semantic**: `Rate 0-10: Does this image show "{prompt}"?`
- **Quality**: `Rate 0-10: How natural is this image? Any artifacts?`
- **Aggregation**: `sqrt(semantic * quality) / 10`
- **Source**: `metric_viescore.py`

### InferBench
- **SC (Semantic/Compliance)**: Long prompt with rules for editing success + overediting
  - Two images (original + edited)
  - `score1` = editing success (0–10), `score2` = overediting (0–10)
  - Output: `[score1, score2]`
- **PQ (Perceptual Quality)**: Long prompt for naturalness + artifacts
  - Single image
  - `naturalness` (0–10), `artifacts` (0–10)
  - Output: `[naturalness, artifacts]`
- **Aggregation**: `min(SC_scores)`, `min(PQ_scores)`, `overall = sqrt(SC * PQ)`
- **Context**: "You are a professional digital artist..." + JSON output format
- **Source**: `viescore.py`

---

## 6. QA Accuracy

### Pruna
- **Question**: `What is in this image? Answer:`
- **Scoring**: 1.0 if non-empty response, else 0.0
- **Use**: Generic image understanding check
- **Source**: `metric_qa_accuracy.py`

### InferBench
- **Questions**: From GenEval metadata (e.g. "Does the image show at least one red apple?", "Does the image show exactly 3 cats?")
- **Template**: `{question} Please answer yes or no.`
- **Expected answers**: `Yes` for all (benchmark-specific)
- **Scoring**: Accuracy over N questions, n_correct, n_incorrect
- **Source**: `qa_accuracy.py`, `geneval.py` (benchmark)

---

## 7. VLM Base Layer (Score Method)

### Pruna – LitellmVLM & TransformersVLM
- **Prompt**: `{question} Please answer yes or no.`
- **Scoring**: `1.0 if answer.lower() in response else 0.0`
- **Scoring**: Same substring check
- **Source**: `vlm_base.py` line 371

### InferBench – OpenAIAPIVLM
- **Scoring**: Prefers logprobs (Yes/No token probabilities) when available
- **Fallback**: Generation + substring check ("yes"/"no" in response)
- **No prompt suffix**: Question passed as-is; metrics add their own suffix
- **Source**: `api_vlm_base.py`

---

## Recommendations

1. **Alignment / VQA**: InferBench’s multi-question + dependency setup is more detailed; Pruna’s single-question version is simpler. For OneIG-style benchmarks, InferBench’s approach is required.

2. **Text Score**: InferBench’s OCR prompt is more explicit and robust; Pruna now uses InferBench-style OCR prompt and supports ground-truth edit distance when gt contains text_content.

3. **Img Edit Score**: InferBench uses full ImgEdit judge prompts; Pruna uses an improved single 0–10 rating with explicit scale instructions. For ImgEdit benchmarks, InferBench’s prompts are necessary.

4. **VieScore**: InferBench’s SC+PQ prompts match the original VieScore design. Pruna’s uses improved explicit 0–10 scale prompts.

5. **VLM Base**: Pruna now uses unified "Please answer yes or no." suffix for both Litellm and Transformers.
