# Linux-GPT: A Transformer Trained on the Linux Kernel

**Project Overview**
A decoder-only Transformer model built from scratch in PyTorch, trained on the Linux Kernel source code to generate syntactically correct C code. This project explores the mechanics of Large Language Models (LLMs), moving from character-level prediction to Byte-Pair Encoding (BPE) tokenization.

**Technical Architecture**
* **Model:** GPT-2 style Transformer (Multi-Head Self-Attention, Feed-Forward Networks).
* **Tokenizer:** OpenAI `tiktoken` (GPT-2 BPE) with a vocab size of 50,304.
* **Dataset:** Raw C source code scraped from the `torvalds/linux` repository.
* **Hardware:** Optimized for CPU training (mps/cpu) with scaled hyperparameters.

**Key Features Implemented**
1.  **Tokenization Upgrade:** Migrated from character-level (96 vocab) to BPE (50k vocab) to improve long-range dependency handling and logic.
2.  **Sampling Strategies:** Implemented Temperature-based sampling (`T > 1.0` vs `T < 1.0`) to solve "repetition loops" common in greedy decoding.
3.  **Context Window:** Managed block size and embeddings to balance memory usage vs. coherence.

**How to Run**
1.  **Install Dependencies:** `pip install torch tiktoken`
2.  **Train the Model:** `python v2_token_model/gpt_token.py`
3.  **Chat with the AI:** `python v2_token_model/chat_token.py`

---

## Phase 3: Fine-Tuning (Llama-3)
* **Model:** Fine-tuned `unsloth/Llama-3.2-3B-Instruct` on the Linux Kernel dataset.
* **Method:** Used LoRA (Low-Rank Adaptation) via the Unsloth library on a Tesla T4 GPU (Google Colab).
* **Results:** The model successfully generates valid Linux Kernel syntax (headers, `static struct file_operations`, `kmalloc`) compared to the base model.
* **Files:** The trained adapters are saved in the `linux_kernel_adapter/` folder.
