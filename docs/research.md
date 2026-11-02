# Open-source offline notes-to-LaTeX tool: research, architecture and build plan

Oct 31, 2026 · @Satyansh

## Summary

Build an open-source desktop app that turns photographed and scanned pages into LaTeX, Markdown and Word on a student laptop, with no upload and no page limit. Under the hood it is not one model with a UI on top. A router sends each region of a page to the cheapest specialist that can read it, checks every output, and escalates only failures to bigger models (see Routing and verification). A paid "fast mode" runs the escalations on rented GPUs.

**The bet.** Model makers such as PaddleOCR, TeleOCR and IndicOCR will keep improving, and we want to benefit from that, not compete with it. We own the layer above them: the router, the checks, the laptop scheduler, the review workflow and the test set of Indian student documents. Every better model they release slots in and makes us better. Running on the student's own laptop makes "unlimited and free" sustainable. Gemini stays better for a single messy page; we win on volume, privacy, verified output and Indian-language documents.

**What v1 is not.** Not handwriting, and not Hindi. TeleOCR's model card covers Chinese and English only and says nothing about handwriting. Both are v2 research tracks, launched only if they pass their own tests.

**Go / no-go gates, in order:**

1. **Week 1, model gate.** On our own 300-page test set, TeleOCR or PaddleOCR-VL must come within 3 points of Gemini Flash on printed and photographed English pages, using the same scoring. Fail → stop.
2. **Week 2, laptop gate.** The quantized model must parse a typical page in 60 seconds or less on a ₹50–60k laptop (Intel Core i5, 16 GB RAM, integrated graphics). Fail → ship as a server-only tool or stop.
3. **Week 8, launch gate.** Twenty real students each convert one real document end to end, and at least 15 say they would use it again instead of Gemini. Fail → fix or stop.
4. **Month 4, handwriting gate.** The fine-tuned model must reach usable accuracy on our handwritten test set before "handwriting" appears anywhere in the marketing.

## Competitive landscape

The market is crowded with single-equation tools and chat assistants. Nobody ships a free, open-source, offline tool for whole documents, so that is the gap we target.

| Option | Price for a student | What it does well | Why a student would still switch |
| --- | --- | --- | --- |
| [Gemini app](https://ai.zenken.co.jp/en/post/gemini-free-vs-pro/) (free tier) | Free, with daily quotas on 3.5 Flash; 10 files per prompt ([source](https://fast.io/resources/gemini-document-upload-limit.md)) | Best accuracy on messy single pages and on real scans of Hindi text | Output is trapped in a chat: copy-paste per page, no document assembly, no check against the original |
| ChatGPT / [OpenAI Prism](https://library.hkust.edu.hk/news-events/news/prism-new-ai-latex-editor-academic-writing) | Free (Prism needs a ChatGPT account) | Converts photos of equations into LaTeX inside an editor | Often skips OCR on scanned PDFs and may answer only from the first pages of long files ([source](https://memx.app/fix/chatgpt-not-reading-uploaded-file/)); silently "fixes" errors ([source](https://arxiv.org/pdf/2603.00895)) |
| [Mathpix Snip](https://mathpix.com/pricing) | 10 PDF pages + 10 images a month free; Pro $4.99/month ([source](https://us.fitgap.com/products/007537/mathpix)) | Accurate and literal; does not guess | The free tier covers a few pages, not a semester |
| [SimpleTex](https://alternativeto.net/software/simpletex) | Free | Accurate equation snips; Word and Typst export | Built for equations, not whole documents; hosted in China |
| [Pix2Text](https://www.file2markdown.ai/blog/mathpix-alternatives) | Free, MIT license, runs locally | Closest open-source option for math | Older models; math only, no full-page workflow |
| [Underleaf](https://chromewebstore.google.com/detail/kkapknfgmppfnheehjfbkcbnmginepcc), [Proofset](https://apps.apple.com/us/app/proofset/id6757517701), [GhostTeX](https://hunted.space/product/ghosttex), [ilovemylatex](https://hunted.space/product/i-love-my-latex) | Freemium, small free quotas | Handwriting-to-LaTeX apps, mostly wrapping cloud LLMs | Small user bases and quotas; GhostTeX is offline but Mac-only and closed |

**Where we win.**

- **No limits, because the student's laptop does the work.** Gemini cannot match this on bulk jobs like a 120-page scanned textbook or a semester of slides.
- **Faithful output.** Every block of output sits beside the crop it came from, flagged when the model is unsure. LLM tools quietly rewrite what they can't read.
- **A finished document.** One LaTeX, Word or Markdown file per input, not a chat transcript.
- **Open source and private.** Nothing leaves the machine, which matters for unpublished research and exam material.

**Where we lose, and say so.** A single messy handwritten page goes to Gemini. Until v2 proves otherwise, our docs and landing page say so.

## Model research

Two small models are good enough to build on, and each wins where the other is weak. TeleOCR leads on photographed English pages; PaddleOCR-VL-1.6 covers Indian scripts, handwriting and llama.cpp. Ship both behind one interface, and let the week-1 test set the default.

| Model | Size, license | Reported strength | Known weakness | Local runtimes |
| --- | --- | --- | --- | --- |
| [TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) (China Telecom, Aug 2026) | \~1.2B, Apache 2.0 | 96.87 on OmniDocBench v1.6 and 88.53 on Wild-OmniDocBench (photos of printed pages), both #1 in its own table | Chinese and English only; handwriting not mentioned; a weak formula score on Dr.DocBench (0.02 vs 0.21 for PaddleOCR-VL); Gemini-3.1-Pro beats it on real degraded scans (71.98 vs 70.85) | Browser via [ONNX/WebGPU](https://huggingface.co/stefanj0/TeleOCR-ONNX), 1.77 GB at q4f16; Rust/Candle port [teleocr-rs](https://huggingface.co/rleo/TeleOCR-GGUF), CPU or CUDA; **not** llama.cpp |
| [PaddleOCR-VL-1.6](https://www.paddleocr.ai/main/en/) (Baidu, May 2026) | 0.9B + a 15.6M layout model, Apache 2.0 | 96.3 on OmniDocBench v1.6; [109 languages](https://blog.roboflow.com/best-open-source-ocr-models/), including Devanagari, Tamil and Telugu; [0.916 on handwritten English](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/) text spotting (v1.5) | The Q4 quantization [hallucinates more](https://developer.lazycat.cloud/aipod/en/benchmark/paddleocr-vl-1.5.html) than Q8 or f16; can loop on long repeated text | [llama.cpp](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/) since build b8110 (CPU, Vulkan, CUDA, Metal); [GGUF](https://huggingface.co/SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF) Q4\_K\_M uses about 1.6 GB of GPU memory |
| [OXR-1.0](https://letsdatascience.com/news/somark-open-sources-13b-oxr-10-document-parser-with-benchmar-903f39c6) (SoMark, Oct 2026) | 1.3B, custom license: free under $2M in revenue or funding, no competing hosted API, attribution required | #1 on three benchmarks in SoMark's own runs; pluggable layout module | [Not independently reproduced](https://fourweekmba.com/ai-open-1-3b-oxr-tops-3-parsing-tests-with-2m-licence-terms/); its OmniDocBench re-test request was still open on Oct 24; the self-hosted setup asks for a 24 GB NVIDIA GPU | GPU servers only |
| GLM-OCR, MinerU2.5-Pro, OvisOCR2 | 0.8–1.2B | 95.2–96.6 on OmniDocBench v1.6 | Nothing that beats the two above for our case | Mixed |

**How OXR fits.** We do not ship it in v1. Its benchmark wins are unverified, it cannot run on a laptop, and its license limits the paid tier. We copy two ideas from it instead:

1. **A swappable layout stage.** Our page pipeline treats layout detection as a separate module, so a notebook-trained layout model can drop in later (see Architecture).
2. **The self-improvement loop.** OXR parses unlabelled pages, has outside models judge and cluster the errors, then generates targeted training data for the next round ([source](https://qbitai.com/2026/10/505635.html)). TeleOCR builds labels from several models voting ([source](https://hyper.ai/ar/papers/2608.12898)). Both recipes are how we get handwriting data cheaply (see Handwriting and Hindi roadmap).

If an independent reproduction confirms OXR's numbers, re-test it as a third engine for the paid GPU tier, after reading its license with a lawyer.

**A competitor to watch.** PaddleOCR's own free API now [parses up to 20,000 pages a day](https://www.paddleocr.ai/latest/en/). That is a cloud service in China, not a student app, but it means "free volume" alone is not a moat. Offline use, privacy and the review workflow are.

## Routing and verification: the moat

We don't wrap one model. Each region of a page goes to the cheapest model that can read it, every output is checked, and only regions that fail the check climb to bigger models. Big open-source parsers already mix models, so routing alone is copyable. Our edge is the checks, laptop-aware scheduling, and drafts that speed up the big model.

&#91;embedded content: routing and verification · 3 tiers, 2 checks\]

Hard regions (handwriting, tables) skip the specialists and go straight to the general model. Everything else tries a specialist first, and anything still unverified is flagged on the review screen rather than guessed.

**What competitors already do.** Hybrid pipelines are now the norm, so "we route between models" is not a pitch:

| Project | How it mixes models | Gap we exploit |
| --- | --- | --- |
| [MinerU](https://newreleases.io/project/github/opendatalab/MinerU/release/mineru-2.7.0-released) (OpenDataLab) | Its default "hybrid" backend combines separate layout, OCR, formula and table models with a VLM | Keeps 4–5 models in memory and [runs out of memory on small GPUs](https://www.nite07.com/en/posts/mineru-low-vram/index.md); no verification step; weak on Indian scripts |
| [Marker](https://pypi.org/project/marker-pdf/) (Datalab) | Surya pipeline, with an optional `--use_llm` pass that sends tables and math to Gemini or another LLM; this lifted a table benchmark [from 0.816 to 0.907](https://www.issoh.co.jp/tech/details/12109/) | The LLM pass is chosen by block type, not by checking the output; the cloud LLM costs money; models are [free only under $2M revenue](https://www.issoh.co.jp/tech/details/12109/) |
| [Docling](https://docling-project.github.io/docling/usage/enrichments/) (IBM) | A pipeline plus optional per-element "enrichment" models for formulas and code, and the 258M Granite-Docling model | Built for enterprise RAG, not students; no review workflow |
| [PaddleOCR-VL](https://arxiv.org/html/2603.24326v2) and [IndicOCR](https://huggingface.co/bodhan-ai/indic-ocr) | Layout model, then one recognizer for every region | One recognizer for everything; no escalation |

**The specialist roster:**

| Region | First choice | Size | Licence | If it fails |
| --- | --- | --- | --- | --- |
| Printed English/Latin lines | [PP-OCRv6](https://blog.roboflow.com/best-open-source-ocr-models/) small or medium | 7.7M–34.5M parameters | Apache 2.0 | General model |
| Formulas | [Texo](https://arxiv.org/html/2602.17189): comparable to the best formula models at 20M parameters, \~80 MB, already runs in a browser ([source](https://discuss.huggingface.co/t/texo-an-in-browser-latex-ocr-model-built-on-transformers-and-transformers-js/169634)); or [PP-FormulaNet\_plus-S](https://www.paddleocr.ai/v3.0.0/en/version3.x/module_usage/formula_recognition.html) | 20M / \~58M | Texo: check the model licence (its web app is [AGPL-3.0](https://awesome.ecosyste.ms/projects/github.com%2Falephpi%2Ftexo-web)); PP-FormulaNet: Apache 2.0 | General model |
| Printed Indian-language text | [IndicOCR](https://huggingface.co/bodhan-ai/indic-ocr) (Bodhan AI): 86.2 on printed Indic text vs 80.4 for Gemini 3.1 Pro, self-reported | 33M layout + 0.8B recognizer | Indic Open Model License: commercial use allowed, share-alike, **written permission needed to offer it as a hosted API** (affects fast mode) | Qwen3-VL-8B in fast mode |
| Handwritten Indian-language text | IndicOCR (experimental): 66.7 vs Gemini 3.1 Pro's 72.0 | Same | Same | Review flag |
| Tables, handwritten English and math | Straight to the general model (PaddleOCR-VL or TeleOCR) | 0.9–1.2B | Apache 2.0 | Fast-mode model, then review flag |

**How each output is checked:**

- **Formulas: render and compare.** Compile the predicted LaTeX, render it, and compare it with the original crop. The CDM metric already scores formulas by rendering LaTeX and matching characters as images ([source](https://arxiv.org/html/2409.03643v1)). We compare against the crop instead of a ground truth, which is our adaptation and untested. LATTE showed that feeding render differences back to a model lifted formula accuracy [from 82.27% to 93.32%](https://www.cs.purdue.edu/homes/lintan/publications/latte-aaai25-poster.pdf) in three rounds.
- **Text: agreement and sanity checks.** Two cheap models agree, or a spot-check by the general model agrees. Characters must be valid for the script, and the output length must match the region size.
- **Calibrate everything.** Raw model confidence scores are often miscalibrated ([source](https://arxiv.org/html/2605.18796v1)), so every threshold is fitted on our test set, not guessed.

**Decide the route first, cascade only on failure.** Picking the model before running anything saved 33–39%, against 21–29% for a try-cheap-first cascade, in a 2026 document-extraction study ([source](https://arxiv.org/pdf/2608.06607)). Region type decides the first model; escalation happens only when a check fails.

**Make the big model faster with drafts.** A February 2026 paper used a lightweight pipeline's output as drafts that the big model checks in parallel. That gave a [2.42× speedup with no accuracy loss](https://www.emergentmind.com/papers/2602.12957) on OmniDocBench for dots.ocr, and up to 4.89× on long documents. The authors promise to release code. Our specialists already produce those drafts, so the general model mostly confirms text instead of writing it.

**Schedule for 8 GB laptops.** The specialists stay loaded, since they total well under 1 GB. The general model loads only when a queue of failed or hard regions builds up, processes them as one batch, then unloads. This is the opposite of MinerU's keep-everything-loaded design.

**What this should buy (a hypothesis until the routing test runs):**

- Speed: the big model sees only failed and hard regions, and runs up to 2–4× faster on them with drafts.
- Price: fast-mode GPUs process only escalated regions, so cost per page drops.
- Trust: every accepted block passed a check, so "verified" means something.
- Upgrades for free: when Paddle, TeleOCR, IIIT or Bodhan AI ship a better model, it slots into the roster. Their progress is our progress.

**Risks.** More models to maintain; a patchwork of licences (IndicOCR's share-alike and hosted-API terms, Texo's AGPL web app); formulas cut in half at region boundaries; and layout detection failing on badly curved photos, where TeleOCR's whole-page mode stays as the fallback.

## Running on student laptops

A typical student laptop should manage 40–90 seconds per dense page with a 1B model. That is fine for an overnight batch, but too slow for live snipping, so v1 is designed around batches. The published speed figures all come from faster hardware than students own, so the week-2 test on real ₹50k laptops decides this.

**The target machine.** Buying guides for 2026 converge on an Intel Core i5 12th/13th gen or Ryzen 5 7000, 16 GB RAM, a 512 GB SSD and integrated graphics as the baseline under ₹50,000 ([source](https://dealsmarketindia.blogspot.com/2026/03/best-laptop-for-coding-students-under.html)). Many budget models still ship with 8 GB ([source](https://www.cashkr.com/articles/top-5-best-student-laptops-under-rs.50000-in-india)). We design for 8 GB RAM and Iris Xe or Radeon integrated graphics, with no NVIDIA card.

**What the evidence says about speed:**

| Measurement | Hardware | Result |
| --- | --- | --- |
| [TeleOCR in the browser](https://huggingface.co/stefanj0/TeleOCR-ONNX), q4f16 | AMD Ryzen AI MAX+ 395, Radeon 8060S (a high-end chip) | 65–90 tokens/s on WebGPU; 57 tokens/s on CPU alone |
| [TeleOCR, teleocr-rs](https://huggingface.co/rleo/TeleOCR-GGUF) | RTX 4090 | \~3.2 s per full page |
| [PaddleOCR-VL, PaddlePaddle on CPU](https://thakicloud.github.io/en/research/paddleocr-vl-09b-multilingual-document-parsing/) | Apple Silicon MacBook, CPU only | 32.65 s per page |
| [PaddleOCR-VL, llama.cpp Q8\_0](https://developer.lazycat.cloud/aipod/en/benchmark/paddleocr-vl-1.5.html) | Vendor device | 2,012 tokens in 11 s |
| [Small models, CPU vs iGPU](https://steveleve.github.io/llm-benchmark-charts.html) | Intel i7-1185G7, Iris Xe | CPU tied or beat the GPU for models of 4B or smaller |
| [3B models on Iris Xe, Vulkan](https://zenvanriel.com/ai-engineer-blog/local-ai-integrated-graphics-vulkan-offload/) | Intel Iris Xe | 15–25 tokens/s |

**Our estimate.** A dense page is about 1,000–2,000 output tokens; the llama.cpp sample above was 2,012. At an assumed 25–40 tokens/s on a student CPU, that is roughly 30–80 s of generation plus a few seconds to encode the image. This is a projection, not a measurement.

**Runtime choice:**

| Runtime | Engine it serves | Pros | Cons | Use in v1 |
| --- | --- | --- | --- | --- |
| llama.cpp (CPU + Vulkan) | PaddleOCR-VL GGUF | One binary for Windows, macOS and Linux; uses Iris Xe and Radeon iGPUs; mature | TeleOCR's Qwen3-style decoder does [not load](https://huggingface.co/rleo/TeleOCR-GGUF); PaddleOCR's layout model needs a separate runtime | Default desktop engine |
| teleocr-rs (Rust/Candle) | TeleOCR | Native Rust, embeds well in a Rust app; full-page pipeline | A one-person community port; no Vulkan | Second desktop engine |
| ONNX Runtime (CPU, DirectML, WebGPU) | TeleOCR ONNX, layout models | Runs the same model in a browser or desktop app; DirectML covers Windows iGPUs | A browser can evict a 1.77 GB model from its cache; WebGPU on Linux is limited | Layout model on desktop; the web demo |
| PaddlePaddle (Python) | Full PaddleOCR-VL pipeline | Official, includes the layout stage | A heavy Python install; poor fit for a consumer app | Reference only, for the test harness |

**Browser support.** WebGPU with 16-bit floats works on 89.8% of Windows machines reporting to web3dsurvey, and on 99.75% of Intel gen-12 integrated GPUs ([source](https://web3dsurvey.com/webgpu/features/shader-f16/platform/Windows)). That is enough for a web demo, but the desktop app stays the main product: it runs long batches in the background, reads folders, and keeps the model on disk.

**8 GB laptops.** Load one engine at a time, use the Q8 quantization (the Q4 version [hallucinates more](https://developer.lazycat.cloud/aipod/en/benchmark/paddleocr-vl-1.5.html)), process one page at a time, and free each page's image after recognition. Both models need about 1.5–2 GB, which leaves room for the operating system and a browser.

## Product definition

v1 is a desktop app that converts a folder of page images or PDFs into one checked LaTeX, Markdown or Word document. Everything runs locally. A paid tier runs the same job on our GPUs in seconds instead of minutes.

**Who it is for first.** Indian engineering and science students who already use LaTeX or Word. Typical jobs: digitize old question papers, turn photographed textbook pages into searchable notes, and convert lecture slides or scanned handouts.

**v1 features (free, open source):**

- Drop in PDFs, images or a folder; batch processing runs in the background with progress shown per page and pause/resume.
- Phone photos are flattened and cleaned before recognition.
- Output: one Markdown file, one compilable .tex file, and a .docx file, plus an image file for each figure.
- **A review screen:** each output block sits beside the crop it came from. Low-confidence blocks are highlighted, and any edit is saved straight back into the document.
- Plain-text search across every converted document on the machine.
- No account, no upload, no page limit. It works offline once the model is downloaded.

**Paid "fast mode" (optional):**

- The same pipeline on rented GPUs: a 100-page batch in a couple of minutes instead of an evening.
- Page credits sold in packs (pricing in Paid tier economics), paid via UPI.
- Pages are deleted from the server after processing; the app tells users exactly what leaves their laptop.
- The free tier is never throttled to push people to pay; fast mode only sells speed.

**Explicitly not in v1:**

- Handwriting and Hindi. The app may still try them, but they are labelled experimental and kept out of marketing until they pass their gate.
- Live snip-to-LaTeX of a single equation; SimpleTex and Mathpix already do this well.
- A phone app; v1 phones use fast mode through a web upload.
- Chat, summaries or "AI tutor" features. We transcribe faithfully, we don't interpret.
- Collaboration and cloud sync. Export opens in lattedit, Overleaf or Word instead.

## Architecture

One Rust core runs an eight-stage page pipeline on the laptop. Every stage is a module behind an interface, so engines can be swapped and the recognition step can be sent to our GPUs without changing anything else.

&#91;embedded content: page pipeline · 8 local stages, 1 optional cloud step\]

Pages flow left to right along the top row, then right to left along the bottom. Only the highlighted step ever leaves the laptop, and only in fast mode.

**Components and decisions:**

| Component | Choice | Why |
| --- | --- | --- |
| App shell | [Tauri 2](https://dev.to/santoshpremi/building-a-private-ai-desktop-app-with-rust-tauri-and-llamacpp-13ha) with a React UI and a Rust core | Smaller installers and lower RAM than Electron, which matters on 8 GB laptops; existing apps already bundle llama.cpp this way |
| Router and engines | A Rust router picks a model per region (see Routing and verification). Specialists (PP-OCRv6, Texo or PP-FormulaNet, IndicOCR) run in-process on ONNX Runtime. The general model runs as a llama-server [sidecar process](https://medium.com/@dillon.desilva/building-local-lm-desktop-applications-with-tauri-f54c628b13d9) (PaddleOCR-VL GGUF), or teleocr-rs for TeleOCR, loaded only when needed. | Every engine sits behind one `recognize(region, task, draft?) → tokens + probabilities` interface, so a better model from anyone slots in. A crashed engine doesn't take down the app. |
| Page cleanup | [UVDoc](https://github.com/tanguymagne/UVDoc) dewarping, already packaged for ONNX as [docuwarp](https://pypi.org/project/docuwarp); OpenCV for crop, deskew and shadow removal | Phone photos are the main source of errors; this runs on CPU in well under a second (to be measured) |
| Layout | PaddleOCR's PP-DocLayoutV3, exported to ONNX | Only the recognition half of PaddleOCR-VL [runs in llama.cpp](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/); the layout half has to be run separately. Keeping it a separate module follows OXR's pluggable-layout idea, so a notebook-trained layout model can replace it later |
| Verification | Formulas: compile with Tectonic, render, compare with the crop. Text: two-model agreement, script-validity and length checks. Plus token probabilities and loop detection, all calibrated on the test set. | Failing a check is what triggers escalation. Small OCR models [loop on repeated text](https://developer.lazycat.cloud/aipod/en/benchmark/paddleocr-vl-1.5.html); anything still failing goes to the review screen. |
| Document model | One JSON file per job: pages → blocks (type, bounding box, source crop, text or LaTeX, confidence, edited flag) | A single source of truth for the review screen and all three exports |
| Export | Markdown written directly; LaTeX from a template and test-compiled with Tectonic; .docx via pandoc | The LaTeX file is checked to compile before the user sees it |
| Storage | SQLite for the job queue and search index; files on disk for images and models | Resumable batches; jobs survive a crash or a laptop closing |
| Model delivery | Downloaded on first run from Hugging Face (with a mirror), SHA-256 verified, \~1.5–2 GB per engine | Keeps the installer under 50 MB |
| Fast-mode server | FastAPI gateway with a queue, then [vLLM](https://huggingface.co/XingChen-AGI/TeleOCR) serving the same two models on rented GPUs; Razorpay/UPI page credits | vLLM already serves both models; nothing new to build in the model layer |
| Web demo | TeleOCR [ONNX on WebGPU](https://huggingface.co/stefanj0/TeleOCR-ONNX) in a static page, one page at a time | Lets people try it from a link on X before installing anything |

**Privacy and telemetry.** No telemetry by default. Users can opt in to anonymous crash reports, and separately to donating pages to the handwriting dataset (see the roadmap). Fast mode shows exactly which page images will be uploaded before each job.

## Test plan

We don't trust anyone's published benchmark, ours included. Every claim in this document is re-tested in two weeks on our own 300 pages and on five real student laptops, before any product code is written.

**1. Build the evaluation set (days 1–4).** 300 pages, each with a hand-checked ground truth in Markdown + LaTeX:

| Bucket | Pages | Source | Why it's in the set |
| --- | --- | --- | --- |
| Printed textbook pages, flatbed scan | 50 | NCERT and university textbooks we own | The easy baseline |
| Printed pages, phone photos (curled, shadowed, angled) | 60 | The same pages re-photographed on 3 cheap phones | Our core claim: photos of print |
| Past exam papers (scanned, often low quality) | 40 | University question-paper archives | A top student use case |
| Lecture slides and handouts | 30 | Course PDFs | Mixed text, tables, figures |
| Math-heavy pages (calculus, linear algebra, physics) | 40 | Textbooks and papers | Tests formula accuracy, a known TeleOCR weak spot |
| Handwritten English notes with math | 50 | Volunteers on the team and their friends | Not a v1 claim; it measures how far v2 has to go |
| Printed Hindi pages | 30 | Hindi-medium textbooks | Checks whether PaddleOCR-VL's Devanagari support is usable |

Labelling cost: about 15–20 minutes a page, pre-filled by the best model and corrected by a person, so roughly 80–100 person-hours. Split across the team, that is a week.

**2. Systems under test.** TeleOCR (PyTorch and teleocr-rs), PaddleOCR-VL-1.6 (PaddlePaddle reference and llama.cpp Q8\_0 and Q4\_K\_M), Gemini Flash through the free app (copy-paste for a 60-page subset), and the Mathpix free tier (10 pages).

**3. Metrics** (the ones [OmniDocBench](https://www.paddleocr.ai/main/en/) uses, so our numbers can be compared with published ones):

- Text: normalized edit distance per block.
- Formulas: CDM or BLEU against the ground-truth LaTeX, plus "compiles in Tectonic" yes/no.
- Tables: TEDS (tree edit distance).
- Reading order: edit distance over block order.
- **Faithfulness:** count of "silent corrections", where output contains content that isn't on the page (a fixed typo, an invented line). Checked by hand on a 50-page sample. This is our differentiator, so we measure it.
- **Flag quality:** does our confidence score catch the bad blocks? Measured as the share of wrong blocks that get flagged, at a flag rate of 10% or less.

**4. Hardware matrix (week 2).**

| Machine | Why |
| --- | --- |
| Intel Core i5-12450H, 16 GB, UHD iGPU | The most-recommended student laptop under ₹50k |
| Intel Core i5-1335U, 16 GB, Iris Xe | A thin-and-light U-series chip |
| AMD Ryzen 5 7530U, 16 GB, Radeon iGPU | The common AMD pick |
| Intel Core i3 or Ryzen 3, 8 GB | The floor; it must work, even if slowly |
| Apple M1/M2 MacBook Air, 8 GB | Common among the X audience we launch to |

For each: seconds per page (median and 90th percentile), peak RAM, whether the laptop stays usable during a batch, and battery drain per 10 pages.

**5. Pass/fail thresholds:**

| Gate | Pass | Fail → action |
| --- | --- | --- |
| Accuracy, printed and photographed pages | Within 3 points of Gemini Flash on text edit distance; formula compile rate 90% or higher | Stop; or ship fast mode only if accuracy is fine but speed fails |
| Speed, 16 GB i5 laptop | Median 60 s/page or less | Investigate Vulkan, smaller image sizes, Q4; if still failing, fast mode becomes the default |
| Memory, 8 GB laptop | Peak under 3.5 GB; no swapping | Ship as "16 GB recommended" |
| Faithfulness | Zero silent corrections on the 50-page sample | Fix decoding settings (greedy decoding, repetition limits) before launch |
| Flags | At least 80% of wrong blocks flagged, at 10% or fewer pages flagged | Tune thresholds; it can launch without this, but the review screen is weaker |

**6. Routing test (week 2, alongside the laptop test).** This decides whether the routing layer is worth building. On the 300 pages, measure:

- The share of regions by type: plain text, formula, table, handwritten, Indian script.
- The share each specialist clears with its check passing, and the specialists' error rate on blocks they accept (must be no worse than the general model's).
- Total time per page for routing plus verification against PaddleOCR-VL on every region, on the i5 laptop.
- Render-and-compare on formulas: how many wrong formulas it catches, and how many correct ones it wrongly rejects.

Pass: specialists clear 60% or more of regions with no loss of accuracy, and the page is at least 2× faster end to end. Fail: ship v1 with the general model only and keep routing as a later optimization.

All results go in the repository as a public benchmark page. A reproducible "we tested it on ₹50k laptops" table is also our launch post.

## Handwriting and Hindi roadmap

Fine-tuning is cheap; the bottleneck is training data we are allowed to use. The best-known handwriting datasets forbid commercial use, so a model trained on them can't power the paid tier. Our own consented student pages become the moat.

**Fine-tuning cost is not the problem.**

- A LoRA fine-tune of PaddleOCR-VL on 2,462 documents took [38 minutes on one NVIDIA L4](https://good23hfaccelerating.pages.dev/naazimsnh02/medocr-vision), peaking at 15.8 GB of GPU memory.
- Baidu's own recipe for teaching it Bengali takes [about 2 hours on one A800](https://ernie.readthedocs.io/en/develop/paddleocr_vl_sft.html).
- A community LoRA on 3,000 samples of Javanese script cut the error rate from [19.7% to 3.2%](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa) (on synthetic test data).
- At ₹50–75 an hour for an L4 in India ([source](https://acecloud.ai/?p=46233)), each experiment costs a few hundred rupees.

**Licences decide which data we can use:**

| Dataset | What it is | Licence | Use it for |
| --- | --- | --- | --- |
| [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | English handwritten text, 657 writers | Non-commercial research only | Research benchmarks only |
| [MathWriting](https://arxiv.org/abs/2404.10690v2) (Google) | 230k human-written + 400k synthetic math expressions with LaTeX | [CC BY-NC-SA 4.0](https://www.arxiv.org/pdf/2404.10690) | Research benchmarks only |
| [IIIT-INDIC-HW-WORDS](https://ilocr.iiit.ac.in/dataset/24) | \~100k handwritten word images per language, 10+ Indian languages | CC BY 4.0 | Production training (with attribution) |
| [IIIT-Indic-HW-UC](https://cvit.iiit.ac.in/usodi/ucciohd.php) | 91k phone-photographed handwritten pages, 13 Indian languages, 1,220 writers | Not stated on the page; ask IIIT-H | Production, if they say yes |
| Our own donated pages | Real student notes, with consent | Ours | Production; the moat |
| Synthetic handwriting | Printed text rendered in handwriting fonts and strokes | Ours | Pre-training warm-up |

Because of these licences, run two tracks. A research model may use everything, for benchmarks and papers. The production model, which ships in the app and the paid tier, uses only permissive and owned data.

**Getting labelled pages cheaply, using the OXR and TeleOCR recipes:**

1. **Collect:** an opt-in "donate this page" button in the app, plus campus drives where volunteers copy a set text (DohaScript's controlled-text design, [531 writers](https://fugumt.com/fugumt/paper_check/2602.18089v1_enmode)), so the ground truth is known in advance.
2. **Pre-label by consensus:** run PaddleOCR-VL, TeleOCR and a large open teacher model (e.g. Qwen3-VL) on each page. Blocks where all agree are accepted; disagreements go to people. Don't use Gemini or GPT outputs as training labels; check their terms first, and default to open models.
3. **Correct:** volunteers fix only the flagged blocks in the same review screen the app already has, so building v1 also builds the labelling tool.
4. **Train:** LoRA on the production track every two weeks; cluster the remaining errors (OXR's self-improvement idea) to decide what data to collect next.

**Targets.** 5,000 English handwritten pages with math, and 3,000 Hindi handwritten pages, by month 4.

**The gate.** On the handwritten buckets of the test set, the production model must come within 5 points of Gemini Flash on edit distance, with zero silent corrections. Only then does "handwriting" appear on the website.

**Hindi is the riskier half.** On real printed Hindi scans the best open model scored [75.2 against Gemini 2.5 Flash's 86.3](https://www.emergentmind.com/papers/2606.29213), and handwriting is harder still. Treat it as a research project that could become a paper and a headline, not a launch promise.

## Paid tier economics

Fast mode should cost us roughly ₹0.02–0.20 a page all-in, so a price of about ₹0.10 a page covers costs as long as the server batches pages. Unbatched serving at the high end would lose money, so fast mode always serves through vLLM with batching. Start on per-second serverless GPUs so idle time costs nothing, and move to reserved GPUs once they would be busy most of the time.

**GPU prices (October 2026):**

| GPU | Where | Price |
| --- | --- | --- |
| RTX 4090 | Indian rental provider | [₹73/hr](https://dev.to/dev_yadav_26252073f3a3761/gpu-rental-india-complete-pricing-guide-2026-4m2k) |
| RTX 4090 | RunPod Community / Secure Cloud | [$0.34 / $0.74 per hr](https://www.hostinger.com/tutorials/?p=158956) |
| RTX 4090 | RunPod serverless, billed only while working | [$1.10 per active hr](https://www.thepricer.org/how-much-does-it-cost-to-rent-an-nvidia-gpu/) |
| NVIDIA L4 | E2E Networks / AceCloud (India) | [₹50/hr / ₹41.92/hr](https://acecloud.ai/?p=46233) |

**Throughput evidence:** PaddleOCR-VL with vLLM runs [2–5 pages/s on one RTX 5090](https://reddit.sentinel-team.org/posts/1sk4y2w/snapshots/2026-04-17T23%3A50%3A43.725322Z) (one user's report) and [1.2 pages/s on an A100](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/) in large batches (the paper). TeleOCR does [3.2 s/page on an RTX 4090](https://huggingface.co/rleo/TeleOCR-GGUF) with no request batching.

**Cost per 1,000 pages on a ₹73/hr RTX 4090 (our arithmetic):**

| Scenario | Pages per hour | GPU cost per 1,000 pages | With 3× overhead (idle time, storage, bandwidth, retries) |
| --- | --- | --- | --- |
| Batched PaddleOCR-VL, \~1 page/s | \~3,600 | \~₹20 | \~₹60 |
| Unbatched TeleOCR, \~0.3 page/s | \~1,100 | \~₹65 | \~₹200 |

**Pricing proposal.** Page credits via UPI, which never expire:

- ₹99 for 1,000 pages (₹0.10 a page).
- ₹249 for 3,000 pages.
- A one-time 100 free fast pages on signup, so people can feel the speed.

For comparison, Mathpix charges [$4.99/month for 1,000 PDF pages and $0.0035 per extra page](https://us.fitgap.com/products/007537/mathpix), and its free tier covers only 10 pages a month.

**When to switch from serverless to reserved.** RunPod's dedicated RTX 4090 costs about 67% of its serverless price per hour ([source](https://www.thepricer.org/how-much-does-it-cost-to-rent-an-nvidia-gpu/)). Once a GPU would be busy more than about two-thirds of the time, reserve it.

**Honest expectation.** Fast mode's job is to cover server costs and fund data labelling, not to be a business. If almost nobody pays, the free app loses nothing; only the server is turned off.

## Build order, risks and open questions

Spend two weeks proving the models and the laptops before writing product code; then six weeks to a beta, four to launch. Handwriting runs in parallel from launch. The plan assumes four engineers.

&#91;embedded content: build order · 4 phases, 4 gates\]

The filled diamonds are the cheap early gates; failing either one ends the project before any real cost.

**Order of work inside "Build v1" (weeks 3–8):**

1. Document JSON schema and the engine interface, so everything else can be built in parallel.
2. Ingest, cleanup and layout modules, tested on the 300-page set.
3. The router and specialists (PP-OCRv6, the formula model, IndicOCR) on ONNX Runtime; then the llama.cpp sidecar for PaddleOCR-VL and teleocr-rs for TeleOCR.
4. Verification (render-and-compare, model agreement) calibrated on the test set; then draft-guided decoding for the general model.
5. Review screen; this doubles as the handwriting labelling tool later.
6. Markdown, LaTeX (compile-checked) and .docx export.
7. Installer, first-run model download, crash recovery.
8. Web demo (TeleOCR on WebGPU) and the public benchmark page.

**Risks:**

| Risk | Mitigation |
| --- | --- |
| TeleOCR's and PaddleOCR-VL's published scores don't hold on Indian documents | Week-1 gate; both are tested against Gemini on our own pages |
| Too slow on ₹50k laptops | Week-2 gate; Vulkan and smaller image sizes; fast mode as the fallback default |
| A community port (teleocr-rs, the ONNX build) is abandoned | Both are open source; we can fork them, and PaddleOCR-VL via llama.cpp needs neither |
| Gemini or ChatGPT ship a polished bulk "PDF to LaTeX" feature | Our moat is offline, unlimited, open and faithful; a cloud feature doesn't erase those |
| Licence problems in training data | Separate research and production training tracks (see the handwriting roadmap) |
| Students upload copyrighted textbooks in fast mode | Delete pages after processing; terms of use put responsibility on the user; never store or share outputs |

**Open questions:**

- [ ] Does IIIT Hyderabad allow production use of IIIT-Indic-HW-UC? (Email the CVIT lab.)
- [ ] Is TeleOCR's 1.2B or PaddleOCR-VL's 0.9B the better default on our set? (Week 1.)
- [ ] Can the PP-DocLayoutV3 layout model be exported cleanly to ONNX, or do we use TeleOCR's built-in layout only?
- [ ] Name, domain and licence for our own code (MIT or Apache 2.0).
- [ ] Ask Bodhan AI for written permission to use IndicOCR in fast mode (its licence requires this for hosted use).
- [ ] Confirm the licence on Texo's model weights; its web app is AGPL-3.0, which we can't bundle.
- [ ] Has the hierarchical speculative decoding code been released, and does it work with PaddleOCR-VL or TeleOCR?

## Sources

All benchmark numbers are self-reported by model authors unless marked otherwise.

**Models**

- [TeleOCR model card](https://huggingface.co/XingChen-AGI/TeleOCR) and [paper summary](https://hyper.ai/ar/papers/2608.12898)
- [TeleOCR ONNX for WebGPU](https://huggingface.co/stefanj0/TeleOCR-ONNX) and [TeleOCR GGUF for teleocr-rs](https://huggingface.co/rleo/TeleOCR-GGUF)
- [PaddleOCR documentation](https://www.paddleocr.ai/main/en/), [PaddleOCR-VL paper](https://arxiv.org/html/2603.24326v2) and [llama.cpp guide](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/)
- [PaddleOCR-VL 1.6 GGUF](https://huggingface.co/SanjeevSOLANKI/PaddleOCR-VL-1.6-GGUF) and [quantization benchmark](https://developer.lazycat.cloud/aipod/en/benchmark/paddleocr-vl-1.5.html)
- OXR: [release report](https://building.it.com/articles/oxr-open-weight-document-parsing-model), [licence and verification check](https://fourweekmba.com/ai-open-1-3b-oxr-tops-3-parsing-tests-with-2m-licence-terms/), [independent summary](https://letsdatascience.com/news/somark-open-sources-13b-oxr-10-document-parser-with-benchmar-903f39c6)

**Competitors and alternatives**

- [Gemini free-tier limits](https://ai.zenken.co.jp/en/post/gemini-free-vs-pro/) and [upload caps](https://fast.io/resources/gemini-document-upload-limit.md)
- [Mathpix pricing](https://us.fitgap.com/products/007537/mathpix)
- [LLM vs Mathpix on handwritten math](https://arxiv.org/pdf/2603.00895) (independent university study)
- [Devanagari OCR stress test](https://www.emergentmind.com/papers/2606.29213) (independent)

**Hardware and runtimes**

- [Student laptop buying guide, 2026](https://dealsmarketindia.blogspot.com/2026/03/best-laptop-for-coding-students-under.html)
- [WebGPU shader-f16 support on Windows](https://web3dsurvey.com/webgpu/features/shader-f16/platform/Windows)
- [Small models on Iris Xe, CPU vs GPU](https://steveleve.github.io/llm-benchmark-charts.html)
- [Tauri + llama.cpp desktop app](https://dev.to/santoshpremi/building-a-private-ai-desktop-app-with-rust-tauri-and-llamacpp-13ha)
- [UVDoc dewarping](https://github.com/tanguymagne/UVDoc)

**Data and fine-tuning**

- [IAM terms of use](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), [MathWriting](https://arxiv.org/abs/2404.10690v2), [IIIT-Indic-HW-UC](https://cvit.iiit.ac.in/usodi/ucciohd.php), [IIIT-INDIC-HW-WORDS](https://ilocr.iiit.ac.in/dataset/24)
- [PaddleOCR-VL LoRA on an L4](https://good23hfaccelerating.pages.dev/naazimsnh02/medocr-vision) and [Baidu's fine-tuning guide](https://ernie.readthedocs.io/en/develop/paddleocr_vl_sft.html)

**Routing, verification and competitors**

- [MinerU hybrid backend](https://newreleases.io/project/github/opendatalab/MinerU/release/mineru-2.7.0-released) and its [memory use on small GPUs](https://www.nite07.com/en/posts/mineru-low-vram/index.md)
- [Marker](https://pypi.org/project/marker-pdf/), its [licence and LLM-mode results](https://www.issoh.co.jp/tech/details/12109/), and [Docling enrichments](https://docling-project.github.io/docling/usage/enrichments/)
- [IndicOCR (Bodhan AI)](https://huggingface.co/bodhan-ai/indic-ocr), [Texo](https://arxiv.org/html/2602.17189), [PP-FormulaNet](https://www.paddleocr.ai/v3.0.0/en/version3.x/module_usage/formula_recognition.html)
- [CDM formula metric](https://arxiv.org/html/2409.03643v1) and [LATTE render-feedback refinement](https://ojs.aaai.org/index.php/AAAI/article/view/32422)
- [Pre-inference routing vs cascades](https://arxiv.org/pdf/2608.06607) and [calibrated cascade routing](https://arxiv.org/html/2605.18796v1)
- [Hierarchical speculative decoding for document parsing](https://arxiv.org/html/2602.12957v1)

**GPU pricing**

- [India GPU rental prices](https://dev.to/dev_yadav_26252073f3a3761/gpu-rental-india-complete-pricing-guide-2026-4m2k), [L4 in India](https://acecloud.ai/?p=46233), [RunPod RTX 4090](https://www.hostinger.com/tutorials/?p=158956), [serverless vs dedicated](https://www.thepricer.org/how-much-does-it-cost-to-rent-an-nvidia-gpu/)
