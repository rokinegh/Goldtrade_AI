# GoldTrade-AI

An NLP-based quality classification system for gold and jewelry e-commerce product listings using transformer embeddings and dimensionality reduction techniques.

## Overview

GoldTrade-AI is a machine learning pipeline designed to analyze and classify product listings in e-commerce datasets, with a specific focus on gold and jewelry items. The project leverages DistilBERT transformer models to generate semantic embeddings and employs t-SNE and UMAP for visualization and quality assessment of product descriptions.

## Features

- **Automated Data Acquisition**: Downloads and processes the Vietnamese Tiki e-commerce dataset via KaggleHub
- **Multilingual Keyword Filtering**: Supports both English and Vietnamese gold/jewelry terminology for comprehensive product detection
- **Transformer-based Embeddings**: Utilizes DistilBERT for extracting [CLS] token embeddings from product descriptions
- **Dimensionality Reduction**: Implements both t-SNE and UMAP for embedding space visualization
- **Quality Metrics**: Computes silhouette scores to quantify cluster separability

## Dataset

The project uses the [Vietnamese Tiki E-commerce Dataset](https://www.kaggle.com/datasets/michaelminhpham/vietnamese-tiki-e-commerce-dataset) from Kaggle, which contains product listings from Tiki, a major Vietnamese e-commerce platform.

### Preprocessing Pipeline

1. **Text Concatenation**: Combines product `name` and `description` fields
2. **Length Filtering**: Removes entries with fewer than 30 characters
3. **Keyword Matching**: Filters products using gold/jewelry-related keywords in both English and Vietnamese
4. **Deduplication**: Removes duplicate entries to ensure data quality

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Dependencies

```bash
pip install kagglehub pandas numpy matplotlib scikit-learn umap-learn transformers torch
```

### Kaggle API Setup

Ensure your Kaggle API credentials are configured:

```bash
# Place your kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Running in Google Colab

1. Open the notebook in Google Colab
2. Run all cells sequentially
3. The pipeline will automatically download the dataset and generate visualizations

### Running Locally

```python
python goldtrade_ai.py
```

## Output

The pipeline generates:

- **Visualization Files**:
  - `figure_4_10_cls_embeddings_tsne_umap.pdf` (600 DPI, publication-ready)
  - `figure_4_10_cls_embeddings_tsne_umap.png` (300 DPI)

- **Console Output**:
  - Dataset statistics and sample entries
  - Silhouette scores for t-SNE and UMAP projections

## Methodology

### Embedding Extraction

The system extracts [CLS] token embeddings from DistilBERT, which capture the semantic representation of entire product descriptions:

```python
def extract_cls_embeddings(texts, batch_size=32):
    # Tokenize and process through DistilBERT
    # Extract [CLS] token from last hidden state
    return embeddings
```

### Dimensionality Reduction

- **t-SNE**: Configured with perplexity=30 (adaptive based on sample size), 1000 iterations
- **UMAP**: Uses n_neighbors=15 (adaptive), min_dist=0.1 for local structure preservation

### Quality Assessment

Silhouette scores measure how well-separated the quality clusters are in the reduced embedding space, with values closer to 1.0 indicating better separation.

## Project Structure

```
GoldTrade-AI/
├── goldtrade_ai.py          # Main pipeline script
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── outputs/
    ├── figure_4_10_cls_embeddings_tsne_umap.pdf
    └── figure_4_10_cls_embeddings_tsne_umap.png
```

## Keywords Supported

### English
`gold`, `karat`, `carat`, `18k`, `22k`, `24k`, `jewelry`, `ring`, `chain`, `necklace`, `bracelet`, `earring`, `pendant`, `ingot`, `bullion`, etc.

### Vietnamese
`vàng`, `nhẫn`, `dây chuyền`, `lắc`, `trang sức`, `bông tai`, `mạ vàng`, etc.

## Future Enhancements

- [ ] Fine-tune DistilBERT on gold/jewelry-specific corpus
- [ ] Implement supervised classification for quality prediction
- [ ] Add support for additional e-commerce platforms
- [ ] Develop API endpoint for real-time product classification
- [ ] Integrate price prediction based on description quality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the DistilBERT implementation
- [Kaggle](https://www.kaggle.com/) for hosting the Tiki e-commerce dataset
- [UMAP](https://umap-learn.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/) for dimensionality reduction tools

## Citation

If you use this project in your research, please cite:

```bibtex
@software{goldtrade_ai,
  title = {GoldTrade-AI: NLP-based Quality Classification for Gold E-commerce Listings},
  year = {2025},
  url = {https://github.com/rokinegh/GoldTrade_AI}
}
```

---

**Note**: This project is for research and educational purposes. Always verify product authenticity through certified dealers when purchasing gold or jewelry.
