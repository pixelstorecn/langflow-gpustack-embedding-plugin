from langchain_openai import OpenAIEmbeddings

from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.base.models.openai_constants import OPENAI_EMBEDDING_MODEL_NAMES
from langflow.field_typing import Embeddings
from langflow.io import BoolInput, DictInput, DropdownInput, FloatInput, IntInput, MessageTextInput, SecretStrInput
from langflow.schema.dotdict import dotdict
from typing import Any
import requests
#gpustack embeddings models plugin in langflow
class GPUStackEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "GPUStack Embeddings"
    description = "Generate embeddings using GPUStack models with GPUStack."
    icon = "GPUStack"
    name = "GPUStackEmbeddings"

    inputs = [
        DictInput(name="default_headers", display_name="Default Headers", advanced=True,
                  info="Default headers to use for the API request."),
        DictInput(name="default_query", display_name="Default Query", advanced=True,
                  info="Default query parameters to use for the API request."),
        IntInput(name="chunk_size", display_name="Chunk Size", advanced=True, value=1000),
        MessageTextInput(name="deployment", display_name="Deployment", advanced=True),
        IntInput(name="embedding_ctx_length", display_name="Embedding Context Length", advanced=True, value=1536),
        IntInput(name="max_retries", display_name="Max Retries", value=3, advanced=True),
        DropdownInput(name="model", display_name="Model", advanced=False, options=[],refresh_button=True,combobox=True),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
        SecretStrInput(name="gpustack_api_key", display_name="GPUStack API Key", value="OPENAI_API_KEY", required=True),
        MessageTextInput(name="gpustack_api_base", display_name="GPUStack API Base", required=True,refresh_button=True),
        MessageTextInput(name="gpustack_api_type", display_name="GPUStack API Type", advanced=True),
        MessageTextInput(name="gpustack_api_version", display_name="GPUStack API Version", advanced=True),
        MessageTextInput(name="gpustack_organization", display_name="GPUStack Organization", advanced=True),
        MessageTextInput(name="gpustack_proxy", display_name="GPUStack Proxy", advanced=True),
        FloatInput(name="request_timeout", display_name="Request Timeout", advanced=True),
        BoolInput(name="show_progress_bar", display_name="Show Progress Bar", advanced=True),
        BoolInput(name="skip_empty", display_name="Skip Empty", advanced=True),
        MessageTextInput(name="tiktoken_model_name", display_name="TikToken Model Name", advanced=True),
        BoolInput(name="tiktoken_enable", display_name="TikToken Enable", advanced=True, value=True,
                  info="If False, you must have transformers installed."),
        IntInput(name="dimensions", display_name="Dimensions", advanced=True,
                 info="The number of dimensions the resulting output embeddings should have. Only supported by certain models."),
    ]
    def fetch_models(self) -> list[str]:
        """Fetch the list of available models from GPUStack API."""
        try:
            response = requests.get(
                f"{self.gpustack_api_base}/models",
                headers={"Authorization": f"Bearer {self.gpustack_api_key}"},
            )
            response.raise_for_status()
            models_data = response.json().get("items", [])
            embedding_models = [
                model["name"]
                for model in models_data
                if "categories" in model and "embedding" in model["categories"] and  model["ready_replicas"] >0
            ]            
            return embedding_models
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching models from GPUStack: {e}")

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Update the build configuration with available models."""
        if field_name in ("gpustack_api_base", "model", "gpustack_api_key") and field_value:
            try:
                models = self.fetch_models()
                if models:
                    build_config["model"]["options"] = models
                    build_config["model"]["value"] = models[0]
            except Exception as e:
                raise ValueError(f"Error getting model names: {e}") from e
        return build_config
        
    def build_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions or None,
            deployment=self.deployment or None,
            api_version=self.gpustack_api_version or None,
            base_url=self.gpustack_api_base or None,
            openai_api_type=self.gpustack_api_type or None,
            openai_proxy=self.gpustack_proxy or None,
            embedding_ctx_length=self.embedding_ctx_length,
            api_key=self.gpustack_api_key or None,
            organization=self.gpustack_organization or None,
            allowed_special="all",
            disallowed_special="all",
            chunk_size=self.chunk_size,
            max_retries=self.max_retries,
            timeout=self.request_timeout or None,
            tiktoken_enabled=self.tiktoken_enable,
            tiktoken_model_name=self.tiktoken_model_name or None,
            show_progress_bar=self.show_progress_bar,
            model_kwargs=self.model_kwargs,
            skip_empty=self.skip_empty,
            default_headers=self.default_headers or None,
            default_query=self.default_query or None,
        )
