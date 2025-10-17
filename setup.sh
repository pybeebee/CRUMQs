
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

uv venv unans_env --python 3.11
source unans_env/bin/activate
uv pip install -r requirements.txt

uv pip install llama-index[cohere,huggingface,bm25,raptor]
uv pip install llama-index-llms-openai
uv pip install llama-index-llms-cohere  llama-index-embeddings-cohere  llama-index-core  llama-index-retrievers
uv pip install llama-index-postprocessor-cohere-rerank
uv pip install llama-index-packs-raptor
uv pip install llama-index-retrievers-bm25
uv pip install llama-index-embeddings-huggingface 
uv pip install llama-index-llms-huggingface
uv pip install llama-index-embeddings-cohere
uv pip install --upgrade llama_index qdrant_client pydantic
uv pip install numpy==1.26.4
uv pip install pydantic==2.9.2
uv pip install llama-index==0.10.22
uv pip install termcolor ipdb paperscraper fitz PyMuPDF langchain-together langchain-anthropic langchain-huggingface
uv pip install json-repair
uv pip install llama-index-llms-openai-like==0.3.0
uv pip install google-generativeai
uv pip install protobuf==6.31.1
uv pip install google-genai
uv pip install --upgrade llama-index-embeddings-openai
uv pip install numpy==1.26.4

echo -e "${GREEN}Environment setup complete!${NC}"

