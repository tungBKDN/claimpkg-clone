"""
FastAPI server for ClaimPKG pipeline.
Singleton pattern for heavy resources to avoid reloading on each request.
"""

import os
import sys
sys.path.append('..')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, final
import uvicorn
from datetime import datetime

from kg_connector.kg_connector import KGConnector
from llm.general_llm import GeneralLLM
from llm.specialized_llm import SpecializedLLM
from llm.basic_sense_llm import BasicSenseLLM
from llm.psedograph_generator_llm import PseudographGeneratorLLM
from middle.group_n_decompose import GroupNDecompose
from middle.retrieve_and_union import RetrieveAndUnion
from middle.greedy import Greedy
from embeddings.embedder import Embedder
from utils.sim import Similarity
from pipeline import Pipeline


# ========================================
# Singleton Registry
# ========================================
class SingletonRegistry:
    """Registry to hold singleton instances of heavy resources."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print("Initializing singleton resources...")
        self.kg_connector = None
        self.general_llm = None
        self.specialized_llm = None
        self.sim = None
        self.embedder = None
        self.group_n_decompose = None
        self.retrieve_and_union = None
        self.pseudograph_generator = None
        self.pipeline = None
        self._initialized = True
        self.init_timestamp = None
        self.basic_sense_llm = None

    def initialize(self):
        """Lazy initialization of all singleton resources."""
        if self.pipeline is not None:
            print("Resources already initialized.")
            return

        print("Loading KGConnector...")
        self.kg_connector = KGConnector()

        print("Loading Similarity encoder...")
        self.sim = Similarity()

        print("Loading Embedder (this may take a while for large KGs)...")
        self.embedder = Embedder(kg_connector=self.kg_connector, sim=self.sim)

        self.sim.kg_entities = self.embedder.kg_entities
        self.sim.entity_embeddings = self.embedder.entity_embeddings
        self.sim.kg_relations = self.embedder.kg_relations
        self.sim.relation_embeddings = self.embedder.relation_embeddings

        print("Loading LLMs...")
        self.general_llm = GeneralLLM()
        # self.specialized_llm = SpecializedLLM()
        self.pseudograph_generator = PseudographGeneratorLLM()
        self.basic_sense_llm = BasicSenseLLM()

        print("Loading GroupNDecompose...")
        self.group_n_decompose = GroupNDecompose(
            embedder=self.embedder,
            kg_connector=self.kg_connector
        )

        print("Loading RetrieveAndUnion...")
        self.retrieve_and_union = RetrieveAndUnion(kg_connector=self.kg_connector)

        print("Initializing Pipeline...")
        # Create pipeline with pre-initialized singletons
        self.pipeline = Pipeline(use_singleton_registry=True)
        self.pipeline.kg_connector = self.kg_connector
        self.pipeline.general_llm = self.general_llm
        # self.pipeline.specialized_llm = self.specialized_llm # This option is currently disabled
        self.pipeline.sim = self.sim
        self.pipeline.embedder = self.embedder
        self.pipeline.group_n_decompose = self.group_n_decompose
        self.pipeline.retrieve_and_union = self.retrieve_and_union
        self.pipeline.basic_sense_llm = self.basic_sense_llm
        self.pipeline.pseudograph_generator = self.pseudograph_generator # This option is currently disabled
        self.pipeline.greedy = Greedy(kg_connector=self.kg_connector)

        self.init_timestamp = datetime.now().isoformat()
        print("✓ All resources initialized successfully!")


# Global singleton registry
registry = SingletonRegistry()


# ========================================
# FastAPI App
# ========================================
app = FastAPI(
    title="ClaimPKG API",
    description="API for claim verification using knowledge graph retrieval",
    version="1.0.0"
)


# ========================================
# Pydantic Models
# ========================================
class ClaimRequest(BaseModel):
    claim: str = Field(..., description="The claim text to verify")
    specialize_mode: str = Field(
        default="FEWSHOT",
        description="Mode for specialized LLM: 'FEWSHOT' or 'FINETUNE'"
    )
    retry: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retries for LLM generation"
    )


class ClaimResponse(BaseModel):
    claim: str
    verdict: Optional[str] = None
    explanation: Optional[str] = None
    final_graph: Optional[str] = None
    time_taken_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    initialized: bool
    init_timestamp: Optional[str] = None
    embeddings: dict


# ========================================
# Startup Event
# ========================================
@app.on_event("startup")
async def startup_event():
    """Initialize all singleton resources on server startup."""
    print("=" * 60)
    print("Starting ClaimPKG API Server...")
    print("=" * 60)
    registry.initialize()
    print("=" * 60)
    print("Server ready to accept requests!")
    print("=" * 60)


# ========================================
# API Endpoints
# ========================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns server status and information about available embeddings.
    """
    initialized = registry.pipeline is not None

    embeddings_info = {
        "relations": {
            "available": False,
            "count": 0,
            "file_path": None
        },
        "entities": {
            "available": False,
            "count": 0,
            "file_path": None
        }
    }

    if initialized and registry.embedder:
        # Check relation embeddings
        if hasattr(registry.embedder, 'relation_txt_file'):
            rel_txt = registry.embedder.relation_txt_file
            if os.path.isfile(rel_txt):
                embeddings_info["relations"]["available"] = True
                embeddings_info["relations"]["file_path"] = rel_txt
                embeddings_info["relations"]["count"] = len(registry.embedder.kg_relations)

        # Check entity embeddings
        if hasattr(registry.embedder, 'entity_txt_file'):
            ent_txt = registry.embedder.entity_txt_file
            if os.path.isfile(ent_txt):
                embeddings_info["entities"]["available"] = True
                embeddings_info["entities"]["file_path"] = ent_txt
                embeddings_info["entities"]["count"] = len(registry.embedder.kg_entities)

    return HealthResponse(
        status="healthy" if initialized else "initializing",
        timestamp=datetime.now().isoformat(),
        initialized=initialized,
        init_timestamp=registry.init_timestamp,
        embeddings=embeddings_info
    )


@app.post("/verify", response_model=ClaimResponse)
async def verify_claim(request: ClaimRequest):
    """
    Verify a claim using the ClaimPKG pipeline.

    - **claim**: The claim text to verify
    - **specialize_mode**: Mode for specialized LLM ('FEWSHOT' or 'FINETUNE')
    - **retry**: Number of retries for LLM generation (1-10)

    Returns retrieved triplets and final answer.
    """
    if registry.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Server is still initializing. Please try again in a moment."
        )

    try:
        import time
        start_time = time.time()

        # Run the pipeline
        final_answer = registry.pipeline.run(
            claim=request.claim,
            specialize_mode=request.specialize_mode,
            retry=request.retry
        )

        processing_time = time.time() - start_time

        print(final_answer)
        return ClaimResponse(
            claim=request.claim,
            verdict=final_answer.get("verdict", None),
            explanation=final_answer.get("explanation", None),
            final_graph=final_answer.get("final_graph", None),
            time_taken_seconds=round(processing_time, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing claim: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ClaimPKG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "verify": "/verify (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ========================================
# Main Entry Point
# ========================================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )
