from typing import Optional

from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.legacy.embeddings import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from huggingface_hub import login

__llm: Optional[HuggingFaceInferenceAPI] = None
__embed_model: Optional[LangchainEmbedding] = None
__token: Optional[str] = None


def __get_token() -> str:
    global __token
    if not __token:
        __token = os.environ["HF_TOKEN"]
        login(token=__token)

    return __token


def get_llm() -> HuggingFaceInferenceAPI:
    return (__llm if __llm
            else
                HuggingFaceInferenceAPI(
                    model_name="HuggingFaceH4/zephyr-7b-alpha",
                    api_key=__get_token()
                )
            )


def get_embed_model() -> LangchainEmbedding:
    return (__embed_model if __embed_model
            else
                LangchainEmbedding(
                    HuggingFaceInferenceAPIEmbeddings(
                        api_key=__get_token(),
                        model_name="thenlper/gte-large"
                    )
                )
            )