import os
from llama_index.core import SimpleDirectoryReader, ServiceContext, StorageContext, VectorStoreIndex, \
    load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from language_models_namespace import get_llm, get_embed_model


def __create_index(service_context: ServiceContext, persist_dir: str) -> VectorStoreIndex:
    index = VectorStoreIndex(SimpleNodeParser()
        .get_nodes_from_documents(
            SimpleDirectoryReader("data").load_data()
        ),
        service_context=service_context,
        storage_context=StorageContext.from_defaults()
    )
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def __load_index(service_context: ServiceContext, persist_dir: str) -> VectorStoreIndex:
    return load_index_from_storage(service_context=service_context,
                                   storage_context=StorageContext.from_defaults(persist_dir=persist_dir)
                                   )


def get_index() -> VectorStoreIndex:
    persist_dir = os.environ["PERSIST_DIR"]

    service_context = ServiceContext.from_defaults(
        llm=get_llm(),
        embed_model=get_embed_model(),
        chunk_size=512)

    return (__load_index(service_context, persist_dir)
            if os.path.exists(persist_dir)
            else __create_index(service_context, persist_dir))
