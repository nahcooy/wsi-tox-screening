from dataclasses import dataclass
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def project_path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else PROJECT_ROOT / value


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    app_env: str = os.getenv("APP_ENV", "development")
    output_dir: Path = project_path_from_env("OUTPUT_DIR", "outputs")
    enable_mock_mode: bool = os.getenv("ENABLE_MOCK_MODE", "true").lower() == "true"
    app_name: str = "WSI Toxicity Screening Workbench"
    trident_root: Path = Path(os.getenv("TRIDENT_ROOT", "/home/nahcooy/MIL/TRIDENT"))
    trident_python: Path = Path(
        os.getenv("TRIDENT_PYTHON", "/home/nahcooy/miniconda3/envs/gg/bin/python")
    )
    default_dataset_csv: Path = Path(
        os.getenv(
            "DEFAULT_DATASET_CSV",
            "/home/nahcooy/MIL/data_checker/260526/case_control_1000each.csv",
        )
    )
    default_wsi_dir: Path = Path(os.getenv("DEFAULT_WSI_DIR", "/mnt/d/spass_2000"))
    mil_lab_root: Path = Path(
        os.getenv("MIL_LAB_ROOT", "/home/nahcooy/MIL/MIL_260527/MIL-Lab")
    )
    mil_python: Path = Path(os.getenv("MIL_PYTHON", "/home/nahcooy/miniconda3/envs/gg/bin/python"))
    default_abmil_checkpoint: Path = project_path_from_env(
        "DEFAULT_ABMIL_CHECKPOINT",
        "models/mil/best_grandqc_univ1_abmil_h5_new_label.pth",
    )
    nulite_root: Path = Path(
        os.getenv("NULITE_ROOT", "/home/nahcooy/NK/NL/NuLite_patch_wise_inference")
    )
    nulite_python: Path = Path(
        os.getenv("NULITE_PYTHON", "/home/nahcooy/miniconda3/envs/cv/bin/python")
    )
    default_nulite_h_checkpoint: Path = project_path_from_env(
        "DEFAULT_NULITE_H_CHECKPOINT",
        "models/nulite/NuLite-H-Weights.pth",
    )
    matched_dataset_csv: Path = project_path_from_env(
        "MATCHED_DATASET_CSV",
        "../tggates_1to1_matched_dataset (2).csv",
    )
    celltype_summary_csv: Path = project_path_from_env(
        "CELLTYPE_SUMMARY_CSV",
        "../summary_celltype_final_260509.csv",
    )
    # Agent (Qwen2.5-VL via Ollama or vLLM)
    agent_base_url: str = os.getenv("AGENT_BASE_URL", "http://localhost:11434")
    agent_model: str = os.getenv("AGENT_MODEL", "qwen2.5vl:7b")
    agent_api_key: str = os.getenv("AGENT_API_KEY", "ollama")
    agent_max_iter: int = int(os.getenv("AGENT_MAX_ITER", "10"))
    agent_timeout_seconds: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "600"))
    # RAG (feedback + literature)
    rag_backend: str = os.getenv("RAG_BACKEND", "json")  # json | faiss | chroma
    rag_data_dir: Path = project_path_from_env("RAG_DATA_DIR", "outputs/rag")
    rag_embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chroma_host: str = os.getenv("CHROMA_HOST", "")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8001"))


settings = Settings()
