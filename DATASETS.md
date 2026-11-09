# Datasets and licences

Every dataset makaar evaluates or trains on, its licence, and what we may use it for. Check this before adding data to any training mix. Not legal advice; confirm the final mix with a lawyer before a paid launch.

## Policy

- **Base model** (PaddleOCR-VL-1.6, PP-DocLayoutV3, PP-OCRv6): Apache 2.0, unchanged.
- **Free-tier handwriting adapter**: a LoRA adapter loaded at runtime on top of the base. It may be trained on non-commercial data, so it is distributed under **CC BY-NC-SA 4.0** (the strictest licence in its mix) with the attributions below. Free and non-commercial only.
- **Paid tier**: needs its own adapter trained only on data marked "commercial use: yes". The base model and app code are unaffected.
- **Evaluation sets are never trained on** (CROHME 2019, MathWriting test, IAM lines, OmniDocBench).
- **No labels from Gemini or other closed models**: their terms bar using outputs to build competing models. Fine as a benchmark.

## Training and evaluation data

| Dataset | Licence | Commercial use | Free-tier adapter | Evaluation | Notes |
|---|---|---|---|---|---|
| [MathWriting](https://huggingface.co/datasets/deepcopy/MathWriting-human) (Google) | CC BY-NC-SA 4.0 | No | Train split | Test split | ShareAlike: the adapter carries CC BY-NC-SA 4.0 |
| [CROHME](https://huggingface.co/datasets/Neeze/CROHME-full) (ICDAR competitions) | Research use; Kaggle copies say CC BY-NC-SA 3.0 | No | Train split | 2019 test | |
| [GNHK](https://github.com/GoodNotes/GNHK-dataset) (GoodNotes) | CC BY 4.0 | Yes, with attribution | Train split | | Real phone photos of English handwriting |
| [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | Non-commercial research | No | **Not used** | Lines (150) | Research-only terms; kept as a clean benchmark |
| [IMGUR5K](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset) (Meta) | CC BY-NC 4.0 | No | Candidate | | |
| [Aida Calculus](https://www.kaggle.com/datasets/aidapearson/ocr-data) (Pearson) | CDLA-Sharing 1.0 | Yes (sharing applies to the data, not trained models) | Candidate | | 100k synthetic handwritten limits |
| [FASTMATH 2025](https://www.kaggle.com/datasets/mssjss/fastmathxassumption-2025-handwritten-math-to-latex) | MIT | Yes | Candidate | | Real, narrow (arithmetic) |
| [25k Math Equation](https://www.kaggle.com/datasets/shiva22btcse0007/25k-math-equation) | MIT | Yes | Candidate | | Synthetic |
| [Handwritten Math Expressions](https://www.kaggle.com/datasets/govindaramsriram/handwritten-math-expressions-dataset) | CC BY 4.0 | Yes, with attribution | Candidate | | Arithmetic |
| [IM2LATEX-230k](https://www.kaggle.com/datasets/gregoryeritsyan/im2latex-230k) | CC0 | Yes | LaTeX source for synthetic data | | Printed |
| [HumynLabs notes](https://huggingface.co/datasets/HumynLabs) (4 sets) | CC BY 4.0 | Yes, with attribution | After labelling | Speed and loop rate | No ground truth |
| [DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) (IBM) | CDLA-Permissive 1.0 | Yes | Printed replay blocks (train shard 0), labelled by the base model itself | | Self-distillation keeps the adapter from forgetting printed text; no human labels used |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Research use | No | Not used | 140 printed pages | |

Not used: HME100K (terms unclear, treated as restricted by others), MathScrub (CC BY-NC 4.0; possible later for the free adapter), OpenWrite and Arabic HME (CC BY-SA 4.0; unclear whether ShareAlike reaches trained models).

## Attribution for the free-tier adapter

Trained in part on MathWriting (Gerth et al., Google, CC BY-NC-SA 4.0), CROHME (ICDAR competitions on recognition of handwritten mathematical expressions), GNHK (GoodNotes, CC BY 4.0) and DocLayNet (Pfitzmann et al., IBM, CDLA-Permissive 1.0).
