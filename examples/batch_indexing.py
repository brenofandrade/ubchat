"""
Exemplo de indexação em lote de múltiplos documentos
"""

import sys
sys.path.append('../')

from src.indexer import DocumentIndexer
from src.indexer.chunking.text_chunker import ChunkStrategy


def main():
    """Exemplo de indexação em lote"""

    # Inicializa o indexador
    indexer = DocumentIndexer(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        use_llm_context=True,
        llm_provider="openai"
    )

    print("=" * 80)
    print("Indexação em Lote com Contexto Enriquecido por LLM")
    print("=" * 80)

    # Indexa todos os documentos
    # Você pode aplicar filtros para processar apenas documentos específicos
    stats = indexer.index_all_documents(
        text_field="content",
        namespace="production",
        limit=None,  # None = todos os documentos
        filters={"status": "pending"},  # Opcional: apenas documentos pendentes
        update_oracle_status=True
    )

    # Exibe estatísticas finais
    print("\n" + "=" * 80)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 80)
    print(f"Total de documentos processados: {stats['total_documents']}")
    print(f"Sucesso: {stats['successful']}")
    print(f"Falhas: {stats['failed']}")
    print(f"Total de chunks criados: {stats['total_chunks']}")
    print(f"Total de vetores inseridos: {stats['total_vectors']}")

    if stats['errors']:
        print("\nErros encontrados:")
        for error in stats['errors']:
            print(f"  - Doc {error['doc_id']}: {error['error']}")

    # Fecha conexões
    indexer.close()


if __name__ == "__main__":
    main()
