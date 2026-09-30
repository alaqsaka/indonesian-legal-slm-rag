# 🧠 Indonesian Legal SLM + Hybrid RAG

Proyek ini menggabungkan dua tahap: **melakukan fine-tuning Small Language Model (SLM)** untuk instruction-following berbahasa Indonesia, lalu **membangun sistem Retrieval-Augmented Generation (RAG)** di atas model tersebut untuk menjawab pertanyaan berdasarkan peraturan hukum Indonesia.

Model hasil fine-tuning tersedia di
[Hugging Face Hub](https://huggingface.co/alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen).
Notebook menyertakan output eksekusi asli agar konfigurasi, proses training, dan
hasil pipeline dapat ditinjau tanpa menjalankan ulang workload GPU.

---

## 📑 Daftar Isi

- [Ringkasan Proyek](#-ringkasan-proyek)
- [Struktur Repositori](#-struktur-repositori)
- [Persyaratan](#️-persyaratan)
- [Fine-Tuning SLM](#1️⃣-fine-tuning-slm)
- [Sistem Hybrid RAG](#2️⃣-sistem-hybrid-rag)
- [Arsitektur Pipeline RAG](#-arsitektur-pipeline-rag)
- [Cara Menjalankan](#-cara-menjalankan)
- [Hasil](#-hasil)
- [Catatan Keamanan](#-catatan-keamanan)

---

## 🎯 Ringkasan Proyek

| Aspek | Keterangan |
|---|---|
| **Tugas** | Question Answering dokumen hukum Indonesia berbahasa Indonesia |
| **Base model** | [`unsloth/Qwen2.5-1.5B`](https://huggingface.co/unsloth/Qwen2.5-1.5B) |
| **Metode fine-tuning** | QLoRA (4-bit) + Unsloth + TRL `SFTTrainer` |
| **Model hasil** | [alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen](https://huggingface.co/alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen) |
| **Korpus RAG** | 4 dokumen PDF peraturan (UU & PP) — total **1.949 halaman** |
| **Vector database** | ChromaDB (lokal) |
| **Strategi retrieval** | Hybrid: **BM25 + Vector Search** → **Cross-Encoder Reranking** |
| **Environment** | Kaggle Notebook / Google Colab (GPU NVIDIA Tesla T4) |

---

## 📂 Struktur Repositori

```
indonesian-legal-slm-rag/
├── notebooks/
│   ├── 01_fine_tune_qwen_indonesian.ipynb
│   └── 02_build_indonesian_legal_rag.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

Notebook dipisahkan berdasarkan urutan eksekusi. Berkas duplikat, ekspor Kaggle
tanpa ekstensi, arsip ZIP, dan metadata sistem operasi sengaja tidak disertakan.

---

## ⚙️ Persyaratan

### Perangkat Keras
- **GPU NVIDIA** dengan VRAM ≥ 12 GB disarankan (proyek ini dijalankan di Tesla T4 14,5 GB).
- Penggunaan CPU memungkinkan tetapi sangat lambat.

### Library Utama
Daftar lengkap ada di `requirements.txt`:

| Kategori | Paket |
|---|---|
| **Fine-tuning** | `unsloth==2026.9.4`, `transformers==4.56.2`, `trl==0.22.2`, `peft==0.20.0`, `bitsandbytes==0.50.2`, `accelerate==1.14.0`, `datasets==4.3.0`, `xformers==0.0.34` |
| **RAG** | `langchain==1.2.3`, `langchain-community==0.4.1`, `langchain-core==1.2.6`, `langchain-huggingface`, `langchain-text-splitters`, `chromadb==1.4.1`, `pymupdf`, `rank-bm25`, `gdown` |
| **Lainnya** | `torch`, `numpy`, `pandas`, `huggingface-hub==0.36.0`, `wandb`, `kagglehub` |

---

## 1️⃣ Fine-Tuning SLM

Fine-tuning dilakukan pada **Qwen2.5-1.5B** menggunakan **QLoRA (Quantized LoRA)** agar dapat berjalan pada GPU dengan VRAM terbatas.

### Konfigurasi Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B",
    max_seq_length=1024,
    load_in_4bit=True,      # kuantisasi 4-bit (QLoRA)
    load_in_8bit=False,
    full_finetuning=False,
)
```

Chat template yang digunakan adalah **`qwen-2.5`**.

### Konfigurasi LoRA

Adapter LoRA dipasang pada **seluruh komponen komputasi utama** model:

```python
model = FastLanguageModel.get_peft_model(
    model,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Multi-Head Attention
        "gate_proj", "up_proj", "down_proj",      # Feed Forward Network
    ],
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

### Dataset

- **Sumber:** [`Ichsan2895/alpaca-gpt4-indonesian`](https://huggingface.co/datasets/Ichsan2895/alpaca-gpt4-indonesian) — **49.969 baris**
- **Format:** diubah ke standar chat template menggunakan fungsi `mapping` dari Hugging Face `datasets`
- **System prompt:**
  > *"Kamu adalah asisten AI yang membantu menjawab pertanyaan pengguna berdasarkan konteks yang diberikan."*

### Hyperparameter Training

Konfigurasi `SFTConfig`:

| Parameter | Nilai |
|---|---|
| `num_train_epochs` | 1 |
| `max_steps` | **800** |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 4 |
| **Total batch size efektif** | 4 |
| `gradient_checkpointing` | `True` |
| `learning_rate` | `2e-4` |
| `lr_scheduler_type` | `linear` |
| `warmup_ratio` | 0.05 |
| `fp16` / `bf16` | `True` / `False` |
| `max_length` | 1024 |
| `packing` | `True` |
| `optim` | `paged_adamw_8bit` |
| `weight_decay` | 0.01 |
| `logging_steps` | 10 |
| `report_to` | `wandb` |
| `save_steps` | 200 (`save_total_limit=2`) |
| `seed` | 3407 |

### Hasil Training

| Metrik | Nilai |
|---|---|
| GPU | 2× NVIDIA Tesla T4 (14,56 GB) |
| Contoh data terpakai | 13.740 |
| Total step | 800 |
| **Parameter terlatih** | **18.464.768 dari 1.562.179.072 (1,18%)** |
| **Peak reserved memory** | **1,97 GB** — tidak terjadi OOM |
| Pelacakan eksperimen | Weights & Biases (WandB) |

### Publikasi Model

Model diunggah ke Hugging Face Hub dengan metode **`merged_16bit`** agar dapat dipanggil kembali pada tahap inferensi RAG:

```python
model.push_to_hub_merged(
    "alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen",
    tokenizer,
    save_method="merged_16bit",
)
```

---

## 2️⃣ Sistem Hybrid RAG

### Korpus Dokumen

4 dokumen PDF peraturan Indonesia yang diunduh otomatis via `gdown` dari Google Drive:

| Dokumen | Jumlah Halaman |
|---|---|
| UU Nomor 6 Tahun 2023 | 1.127 |
| PP Nomor 5 Tahun 2021 | 739 |
| PP Nomor 35 Tahun 2021 | 56 |
| PP Nomor 51 Tahun 2023 | 27 |
| **Total** | **1.949 halaman** |

Dokumen dimuat menggunakan `PyMuPDFLoader`.

### Text Splitting (Parent-Child)

Menggunakan `RecursiveCharacterTextSplitter` dengan dua tingkat pemotongan:

```python
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)
```

Potongan **child** (400 karakter) digunakan untuk pencarian presisi, sedangkan potongan **parent** (2000 karakter) diberikan ke LLM sebagai konteks agar informasi tetap utuh.

### Embedding & Vector Store

- **Model embedding:** `BAAI/bge-m3` (open-source, dijalankan di `cuda`)
- **Vector database:** **ChromaDB** lokal — collection `hukum_indo`
- **Doc store:** `InMemoryByteStore`
- **Retriever:** `ParentDocumentRetriever` dengan `search_type="similarity"`, `k=5`

### Hybrid Retrieval

Menggabungkan pencarian **leksikal (BM25)** dan **semantik (vektor)** untuk menutupi kelemahan masing-masing:

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, retriever],
    weights=[0.5, 0.5]
)
```

- `bm25_retriever` — `BM25Retriever.from_documents(all_documents)`, `k=5`
- `retriever` — `ParentDocumentRetriever` (vektor + doc store)

### Reranking

Hasil hybrid retrieval disaring ulang menggunakan **cross-encoder**:

```python
reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=hybrid_retriever
)
```

Hanya **3 dokumen teratas** yang lolos ke tahap generasi jawaban.

### Model Generator

Menggunakan model hasil fine-tuning dari tahap sebelumnya:

```python
model_name = "alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=4096,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Parameter generasi: `temperature=0.2`, `do_sample=True`, `repetition_penalty=1.2`, `max_new_tokens=512`.

### Prompt Template

Template bergaya **Llama-3** yang menginstruksikan model untuk **hanya menjawab berdasarkan konteks** dan **jujur bila jawaban tidak ditemukan**, serta mencantumkan sumber:

```
Anda adalah asisten AI hukum yang membantu menjawab pertanyaan pengguna
berdasarkan dokumen hukum Indonesia.
Gunakan hanya informasi yang tersedia pada konteks berikut untuk menjawab pertanyaan.
Jika jawaban tidak ditemukan dalam konteks tersebut, sampaikan dengan jujur
bahwa Anda tidak mengetahui jawabannya.
Berikan jawaban secara singkat, jelas, dan cantumkan sumber dokumennya.
```

---

## 🔄 Arsitektur Pipeline RAG

```
                    Pertanyaan pengguna
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   BM25Retriever                        ParentDocumentRetriever
   (leksikal, k=5)                      (vektor bge-m3 + doc store)
        │                                       │
        └───────────► EnsembleRetriever ◄───────┘
                     weights = [0.5, 0.5]
                            │
                            ▼
              CrossEncoderReranker (bge-reranker-base)
                        top_n = 3
                            │
                            ▼
              ContextualCompressionRetriever
                            │
                            ▼
                  PromptTemplate (Llama-3 style)
                            │
                            ▼
        LLM: alaqsaka/fine-tuning-slm-legal-chatbot-using-qwen
             (Qwen2.5-1.5B + LoRA adapter, max_seq_length=4096)
                            │
                            ▼
                    Jawaban + sumber dokumen
```

Rantai ini dibangun dengan **LangChain Expression Language (LCEL)**:

```python
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 🚀 Cara Menjalankan

### 1. Clone repositori

```bash
git clone git@github.com:alaqsaka/indonesian-legal-slm-rag.git
cd indonesian-legal-slm-rag
```

### 2. Install dependensi

```bash
pip install -r requirements.txt
```

> Untuk fine-tuning, install `unsloth` sesuai instruksi resmi di [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) agar cocok dengan versi CUDA Anda.

### 3. Jalankan notebook

| Notebook | Isi | Cara menjalankan |
|---|---|---|
| `notebooks/01_fine_tune_qwen_indonesian.ipynb` | Fine-tuning Qwen2.5-1.5B | Kaggle Notebook / Google Colab dengan GPU |
| `notebooks/02_build_indonesian_legal_rag.ipynb` | Membangun & menguji sistem RAG | Kaggle Notebook / Google Colab dengan GPU |

### 4. Mode interaktif

Notebook RAG diakhiri dengan **mode interaktif** di terminal:

```
============================================================
RAG Hukum Indonesia - Interactive Mode (v6)
============================================================
Dokumen: UU No.6/2023, PP No.35/2021, PP No.51/2023, PP No.5/2021
Ketik 'exit' atau 'quit' untuk keluar

📝 Masukkan pertanyaan hukum: apakah kalau saya lembur akan dibayar lebih?
```

---

## 📊 Hasil

Pipeline RAG berhasil dibangun dan dijalankan end-to-end: mulai dari pengunduhan 4 PDF (1.949 halaman), pemotongan teks parent-child, pembuatan vector store ChromaDB dengan embedding `bge-m3`, hybrid retrieval BM25 + vektor, reranking `bge-reranker-base`, hingga generasi jawaban oleh model hasil fine-tuning.

Contoh pertanyaan yang diuji: **"apakah kalau saya lembur akan dibayar lebih?"** — dijawab dengan konteks dari dokumen ketenagakerjaan yang tersedia.

---

## 🔒 Catatan Keamanan

- **Jangan pernah menuliskan token Hugging Face secara langsung (hardcoded) di dalam notebook.** Gunakan secrets/environment variable:
  ```python
  # Google Colab
  from google.colab import userdata
  token = userdata.get("HF_TOKEN")

  # atau environment variable
  import os
  token = os.environ["HF_TOKEN"]
  ```
- Token dengan akses tulis dapat dipakai orang lain untuk memodifikasi repositori model Anda bila bocor.
- Notebook dalam repositori ini membaca `HF_TOKEN` dari environment dan tidak
  menyimpan nilai token. Jika sebuah token pernah ter-commit, hapus atau rotasi
  token tersebut melalui pengaturan akun Hugging Face; menghapus teks dari commit
  terbaru tidak membatalkan token yang sudah terekspos.

---

## 👤 Penulis

**Al-Aqsa Krisnaya Abidin**
- GitHub: [@alaqsaka](https://github.com/alaqsaka)
- Hugging Face: [@alaqsaka](https://huggingface.co/alaqsaka)
