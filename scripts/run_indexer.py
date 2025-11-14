#!/usr/bin/env python3
"""
Script CLI para executar o indexador

Uso:
    python run_indexer.py --all                    # Indexa todos os documentos
    python run_indexer.py --doc-id 123             # Indexa documento específico
    python run_indexer.py --search "query"         # Busca documentos
    python run_indexer.py --stats                  # Mostra estatísticas
"""

import argparse
import sys
sys.path.append('../')

from src.indexer import DocumentIndexer
from src.indexer.chunking.text_chunker import ChunkStrategy


def index_all(args):
    """Indexa todos os documentos"""
    indexer = DocumentIndexer(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        use_llm_context=not args.no_llm,
        llm_provider=args.llm_provider
    )

    stats = indexer.index_all_documents(
        text_field=args.text_field,
        namespace=args.namespace,
        limit=args.limit
    )

    print("\n✅ Indexação concluída!")
    print(f"   Sucesso: {stats['successful']}/{stats['total_documents']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   Vetores: {stats['total_vectors']}")

    indexer.close()


def index_document(args):
    """Indexa um documento específico"""
    indexer = DocumentIndexer(
        chunk_strategy=ChunkStrategy.RECURSIVE,
        use_llm_context=not args.no_llm,
        llm_provider=args.llm_provider
    )

    result = indexer.index_document(
        doc_id=args.doc_id,
        text_field=args.text_field,
        namespace=args.namespace
    )

    print("\n✅ Documento indexado!")
    print(f"   Doc ID: {result['doc_id']}")
    print(f"   Chunks: {result['chunks']}")
    print(f"   Vetores: {result['vectors_upserted']}")

    indexer.close()


def search(args):
    """Busca documentos"""
    indexer = DocumentIndexer()

    results = indexer.search(
        query=args.search,
        top_k=args.top_k,
        namespace=args.namespace
    )

    print(f"\n🔍 Resultados para: '{args.search}'\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.4f}")
        print(f"   Doc: {result['metadata']['doc_id']} (Chunk {result['metadata']['chunk_index']})")
        print(f"   Tópico: {result['metadata'].get('topic', 'N/A')}")
        print(f"   {result['metadata']['text'][:150]}...")
        print()

    indexer.close()


def show_stats(args):
    """Mostra estatísticas do índice"""
    indexer = DocumentIndexer()

    stats = indexer.get_stats(namespace=args.namespace)

    print("\n📊 Estatísticas do Índice\n")
    print(f"   Namespace: {args.namespace or 'default'}")
    print(f"   Total de vetores: {stats.get('total_vector_count', 0)}")
    print(f"   Dimensões: {stats.get('dimension', 'N/A')}")

    if 'namespaces' in stats:
        print("\n   Namespaces:")
        for ns, ns_stats in stats['namespaces'].items():
            print(f"     - {ns}: {ns_stats.get('vector_count', 0)} vetores")

    indexer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Indexador de documentos com contexto enriquecido por LLM"
    )

    # Comandos principais
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Indexa todos os documentos")
    group.add_argument("--doc-id", type=int, help="Indexa documento específico")
    group.add_argument("--search", type=str, help="Busca documentos")
    group.add_argument("--stats", action="store_true", help="Mostra estatísticas")

    # Opções comuns
    parser.add_argument("--namespace", default="", help="Namespace do Pinecone")
    parser.add_argument("--text-field", default="content", help="Campo do texto no Oracle")
    parser.add_argument("--no-llm", action="store_true", help="Desativa enriquecimento com LLM")
    parser.add_argument("--llm-provider", choices=["openai", "anthropic"], default="openai")

    # Opções específicas
    parser.add_argument("--limit", type=int, help="Limite de documentos (--all)")
    parser.add_argument("--top-k", type=int, default=5, help="Número de resultados (--search)")

    args = parser.parse_args()

    try:
        if args.all:
            index_all(args)
        elif args.doc_id:
            index_document(args)
        elif args.search:
            search(args)
        elif args.stats:
            show_stats(args)

    except Exception as e:
        print(f"\n❌ Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
