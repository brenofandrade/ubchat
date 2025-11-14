"""
Exemplo básico de indexação de documentos com contexto enriquecido por LLM
"""

import sys
sys.path.append('../')

from src.indexer import DocumentIndexer
from src.indexer.chunking.text_chunker import ChunkStrategy


def main():
    """Exemplo básico de uso do indexador"""

    # 1. Inicializa o indexador
    # Por padrão, carrega configurações do arquivo .env
    indexer = DocumentIndexer(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        use_llm_context=True,  # Ativa o enriquecimento com LLM
        llm_provider="openai"  # Pode ser "openai" ou "anthropic"
    )

    # 2. Indexa um documento específico
    print("=" * 80)
    print("Indexando documento único...")
    print("=" * 80)

    result = indexer.index_document(
        doc_id=1,  # ID do documento no Oracle
        text_field="content",  # Nome do campo com o texto
        namespace="production",  # Namespace opcional no Pinecone
        update_oracle_status=True  # Atualiza status no Oracle
    )

    print(f"\nResultado:")
    print(f"  - Chunks criados: {result['chunks']}")
    print(f"  - Vetores inseridos: {result['vectors_upserted']}")
    print(f"  - Total de tokens: {result['total_tokens']}")
    print(f"  - LLM Context usado: {result['use_llm_context']}")

    # 3. Busca por similaridade
    print("\n" + "=" * 80)
    print("Testando busca...")
    print("=" * 80)

    query = "Como funciona o sistema de autenticação?"
    results = indexer.search(
        query=query,
        top_k=5,
        namespace="production"
    )

    print(f"\nResultados para: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Doc ID: {result['metadata']['doc_id']}")
        print(f"   Chunk: {result['metadata']['chunk_index']}")
        print(f"   Tópico: {result['metadata'].get('topic', 'N/A')}")
        print(f"   Resumo: {result['metadata'].get('contextual_summary', 'N/A')[:100]}...")
        print(f"   Texto: {result['metadata']['text'][:150]}...")

    # 4. Obtém estatísticas
    stats = indexer.get_stats(namespace="production")
    print(f"\nEstatísticas do índice:")
    print(f"  - Total de vetores: {stats.get('vector_count', 'N/A')}")

    # 5. Fecha conexões
    indexer.close()


if __name__ == "__main__":
    main()
