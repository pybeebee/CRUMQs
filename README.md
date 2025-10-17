# Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries

Existing RAG benchmarks rarely reflect realistic task complexity for multi-hop or out-of-scope questions, which often can be cheated via disconnected reasoning or require only simple factual recall. This limits the ability for such benchmarks to uncover limitations of existing RAG systems. 

This repository provides code for the first pipeline for automatic, difficulty-controlled creation of un<u>c</u>heatable, <u>r</u>ealistic, <u>u</u>nanswerable, and <u>m</u>ulti-hop <u>q</u>uerie<u>s</u> (CRUMQs), adaptable to any corpus and domain. It includes code to (1) create CRUMQs over two popular RAG datasets and (2) evaluate the resulting benchmark's effectiveness via experiments on leading retrieval-augmented LLMs. 

Compared to prior RAG benchmarks, CRUMQs are highly challenging for RAG systems and achieve up to 81.0% reduction in cheatability scores. More broadly, our pipeline offers a simple way to enhance benchmark difficulty and realism and drive development of more capable RAG systems.

<p align="center">
    <a href="https://www.arxiv.org/abs/2510.11956" style="display:inline-block;background-color:#2196F3;color:white;padding:10px 20px;text-align:center;text-decoration:none;font-size:20px;border-radius:5px;">📄 <b>Paper</b></a>
</p>

## Quick Links

- [🛠 Setup & Installation](#installation)
- [📂 File Structure](#structure)
- [📊 Usage & Benchmarking](#usage)
- [📝 Citation](#citation)

<a name="installation"></a> 

## 🛠 Setup & Installation

1. **Clone the Repository**

2. **Create Environment & Install Dependencies**
    ```bash
    cd CRUMQs
    chmod +x setup.sh && setup.sh
    source unans_env/bin/activate
    export PYTHONPATH=.
    export TOKENIZERS_PARALLELISM=true
    ```
3. **Configure Environment Variables**\
    Create a .env file (or set these in your shell) with the following keys:
    * `OPENAI_API_KEY`: To access GPT models
    * `GEMINI_API_KEY`: To access Gemini models
    * `COHERE_API_KEY`: To access Cohere models
    * `LITELLM_API_KEY`: To access additional models
    * `LITELLM_IP`: LiteLLM server address
    
    We use LiteLLM for inference with open-source models.

4. **Dataset Preparation**\
    To prepare the relevant files from the [TREC NeuCLIR 2024](https://ir-datasets.com/neuclir.html) and [TREC RAG 2025](https://trec-rag.github.io/annoucements/2025-rag25-corpus) datasets for CRUMQs generation and benchmarking, run the following:
    ```bash
    cd CRUMQs
    tar -xvf _database.tar
    tar -xvf _benchmark_database.tar
    ```


<a name="structure"></a> 


## 📂 File Structure

- `_benchmark_database/`: Directory containing database for benchmarking experiments.
- `_database/`
  - `neuclir_test/`: Directory containing golden retrieved articles per test topic for TREC NeuCLIR 2024.
  - `trecrag2025/`: Directory containing golden retrieved articles per test topic for TREC RAG 2025.
  - `test_requests_neuclir.jsonl`: Test topics for TREC NeuCLIR 2024.
  - `test_requests_trecrag2025.jsonl`: Test topics for TREC RAG 2025.
- `src/`: Core benchmark generation & evaluation code.
  - `crumqs_generation/`: Code to generate CRUMQs.
    - `prompts/`: Prompts used in the CRUMQs pipeline.
    - `crawler.py`: Code to scrape external news articles.
    - `crawl_science.py`: Code to scrape external scientific articles.
    - `create_dataset.py`: Main code to create CRUMQs.
    - `dataset_format.py`: Code for Pydantic representation of questions and answers.
    - `filter_dataset.py`: Code to filter & compile resulting CRUMQs generated over the two document collections.
    - `rag.py`: Code to help verify CRUMQs unanswerability.
    - `run.py`: Code to run CRUMQs creation.
    - `run*.sh`: Sample run scripts for generating CRUMQs over the NeuCLIR & RAG collections.
    - `utils*.py`: Utility functions for various parts of the pipeline. 
  - `evaluation/`: Code to evaluate RAG systems on CRUMQs and related benchmarks.
    - `prompts_evaluation.py`: Prompts to score RAG system responses.
    - `prompts_generation.py`: Prompts to elicit retrieval-augmented LLM responses.
    - `run_rag.py`: Code to implement and run RAG systems.
    - `score_rag.py`: Code to performance of RAG systems using different generator models.
- `requirements.txt`: Dependencies for environment creation.
- `setup.sh_.txt`: Script for environment creation.


<a name="usage"></a>

## 📊 Usage & Benchmarking

We provide is a simple pipeline for automatic generation of un<u>c</u>heatable, <u>r</u>ealistic, <u>u</u>nanswerable, and <u>m</u>ulti-hop <u>q</u>uerie<u>s</u> (CRUMQs) that are robust against reasoning shortcuts, target content beyond models' training data cutoff dates, and can be tailored to any document corpus.

![CRUMQs Generation Pipeline](./__figs/steps.png)

In particular, we leverage recent insights in synthetic data generation to ensure coverage of diverse task types and complexity levels, with benchmark difficulty easily controllable via the distribution of in- vs. out-of-scope hops per question.


### 🔧 Generating CRUMQs
To create CRUMQs over the TREC NeuCLIR 2024 and TREC RAG 2025 datasets, simply run the following:

```bash
cd CRUMQs

# Generate CRUMQs
bash ./src/crumqs_generation/run_neuclir.sh
bash ./src/crumqs_generation/run_trecrag.sh

# Compile & Filter CRUMQs
bash ./src/crumqs_generation/filter_dataset.py
```

Detailed of the data generation parameters are described in [run.py](https://github.com/pybeebee/CRUMQs/blob/4bd7eff3556fe23dbb5dbdbe811ddf70862cd52b/src/crumqs_generation/run.py) and may be altered via command-line arguments as desired (e.g., the number of total queries generated, or the number of documents to consider when generating question-answer pairs).

The resulting CRUMQs will be automatically saved, compiled, and assigned quality scores via LLM judgment. Scoring thresholds may be adjusted in [filter_dataset.py](https://github.com/pybeebee/CRUMQs/blob/4bd7eff3556fe23dbb5dbdbe811ddf70862cd52b/src/crumqs_generation/filter_dataset.py).


### 🔍 Evaluating RAG Systems

Use [run_rag.py](https://github.com/pybeebee/CRUMQs/blob/4bd7eff3556fe23dbb5dbdbe811ddf70862cd52b/src/evaluation/run_rag.py) and [score_rag.py](https://github.com/pybeebee/CRUMQs/blob/4bd7eff3556fe23dbb5dbdbe811ddf70862cd52b/src/evaluation/score_rag.py) to configure and benchmark different RAG systems on the final CRUMQs. For example:

```bash
cd CRUMQs; source ./unans_env/bin/activate
export PYTHONPATH=.; export TOKENIZERS_PARALLELISM=true

### RAG System: GPT-5 generator, OpenAI embedding, & Vector retriever

# Get predictions
python ./src/evaluation/run_rag.py  --database_dir=./_database/_benchmark_database  --results_dir=./_predictions  --dataset_path=./generated_data/filtered_crumqs.json  --evaluation_mode=gateway  --generator_llm=gpt-5 --embedding=openai --retrieval=vector

# Score predictions
python ./src/evaluation/score_rag.py  --gt_data_path=./generated_data/filtered_crumqs.json  --results_path=./_predictions/gateway/filtered_crumqs/gpt-5__emb_openai__ret_vector__rerank_None__rewr_None__prompt_DEFAULT/predictions.json

### RAG System: GPT-4o generator, Cohere embedding, Vector retriever, Cohere reranker, & HyDE rewriter

# Get predictions
python ./src/evaluation/run_rag.py  --database_dir=./_database/_benchmark_database --results_dir=./_predictions  --dataset_path=./generated_data/filtered_crumqs.json  --evaluation_mode=rag  --embedding=cohere  --retrieval=vector  --reranker=cohere  --rewriting=hyde

# Score predictions
python ./src/evaluation/score_rag.py  --gt_data_path=./generated_data/filtered_crumqs.json  --results_path=./_predictions/rag/filtered_crumqs/gpt-4o__emb_cohere__ret_vector__rerank_cohere__rewr_hyde__prompt_DEFAULT/predictions.json
```

Evaluation metrics are defined consistently with [prior work](https://arxiv.org/abs/2412.12300). Scores for each RAG system are automatically saved to the same directory as the predictions.


<a name="citation"></a> 

## 📝 Citation

If you find the content of this project helpful, please cite our paper as follows:
```bibtex
@article{liu2025crumqs,
    title={Evaluating Retrieval-Augmented Generation Systems on Unanswerable, Uncheatable, Realistic, Multi-hop Queries}, 
    author={Gabrielle Kaili-May Liu and Bryan Li and Arman Cohan and William Gantt Walden and Eugene Yang},
    journal={arXiv preprint arXiv:2510.11956},
    year={2025},
    url={https://arxiv.org/abs/2510.11956}, 
}
```