from pathlib import Path

import os

AUTO_INDEX = os.environ.get("AUTO_INDEX", "false").lower() == "true"
SOURCE_DIRS = [
    "/media/sampk/350GB/1_qpivolta/QpiVOLTA_papers",
    "/media/sampk/512GB/uv_envs/ollama_rag/conversations",
    "/media/sampk/512GB/uv_envs/ollama_rag/downloaded_papers",
    "/media/sampk/512GB/data_for_rag",
    "/media/sampk/512GB/Textbooks",
]
DB_DIR = "/media/sampk/512GB/uv_envs/ollama_rag/qpi_research_db"
CONV_DIR = "/media/sampk/512GB/uv_envs/ollama_rag/conversations"
DOWNLOAD_DIR = "/media/sampk/512GB/uv_envs/ollama_rag/downloaded_papers"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".cpp",
    ".c",
    ".h",
    ".java",
    ".md",
    ".pptx",
    ".docx",
    ".csv",
    ".xlsx",
    ".json",
    ".f",
    ".f90",
    ".f95",
    ".f77",
    ".for",
    ".ftn",
}

AVAILABLE_MODELS = {
    "1": ("granite4:3b", "fast, fits in VRAM fully"),
    "2": ("phi4-mini", "fast, good reasoning"),
    "3": ("gemma4:e4b", "slow but smartest, partial CPU"),
    "4": ("gemma4:e2b", "medium speed, good quality"),
    "5": ("qwen2.5-coder:7b", "for coding and learning coding"),
    "6": ("qwen3:8b", "for coding and learning coding"),
}

EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "qpi_papers"

# Create dirs on import
for d in [CONV_DIR, DOWNLOAD_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

WORKSPACE_DIR = "/media/sampk/512GB/uv_envs/ollama_rag/workspace"
Path(WORKSPACE_DIR).mkdir(parents=True, exist_ok=True)
