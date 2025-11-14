"""
Exemplo de busca semântica avançada
"""

import sys
sys.path.append('../')

from src.indexer import DocumentIndexer


def main():
    """Exemplo de busca semântica"""

    # Inicializa o indexador
    indexer = DocumentIndexer()

    print("=" * 80)
    print("Busca Semântica Avançada")
    print("=" * 80)

    # Lista de queries para testar
    queries = [
        "Como funciona a autenticação de usuários?",
        "Política de privacidade e dados",
        "Processo de integração com APIs externas",
        "Requisitos técnicos do sistema"
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")

        # Busca básica
        results = indexer.search(
            query=query,
            top_k=3,
            namespace="production"
        )

        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Documento: {result['metadata']['doc_id']}")
            print(f"   Chunk: {result['metadata']['chunk_index']}")

            # Informações enriquecidas pelo LLM
            print(f"\n   📌 Tópico: {result['metadata'].get('topic', 'N/A')}")
            print(f"   📝 Resumo: {result['metadata'].get('contextual_summary', 'N/A')}")
            print(f"   🔑 Conceitos: {result['metadata'].get('key_concepts', 'N/A')}")
            print(f"   🏷️  Keywords: {result['metadata'].get('keywords', 'N/A')}")
            print(f"   ❓ Perguntas: {result['metadata'].get('questions', 'N/A')}")

            print(f"\n   📄 Texto:")
            print(f"   {result['metadata']['text'][:200]}...")

    # Busca com filtros
    print(f"\n{'=' * 80}")
    print("Busca com Filtros de Metadata")
    print(f"{'=' * 80}")

    filtered_results = indexer.search(
        query="configuração do sistema",
        top_k=5,
        namespace="production",
        filters={
            "topic": {"$eq": "Configuração"}
        }
    )

    print(f"\nEncontrados {len(filtered_results)} resultados filtrados")

    # Fecha conexões
    indexer.close()


if __name__ == "__main__":
    main()
