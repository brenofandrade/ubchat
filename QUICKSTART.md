# Quickstart - UBChat Indexador

Guia rápido para começar a usar o sistema de indexação com contexto enriquecido por LLM.

## 1. Instalação Rápida

```bash
# Clone o repositório
git clone <repository>
cd ubchat

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env com suas credenciais
```

## 2. Configure o Banco de Dados

```bash
# Execute o script SQL no seu Oracle Database
sqlplus user/password@host:port/service @scripts/setup_database.sql
```

Isso criará:
- Tabela `documents` com campos necessários
- Índices para performance
- Dados de exemplo (opcional)

## 3. Primeiro Uso - Indexar um Documento

```python
from src.indexer import DocumentIndexer

# Cria o indexador
indexer = DocumentIndexer(
    use_llm_context=True,  # Ativa enriquecimento com LLM
    llm_provider="openai"  # ou "anthropic"
)

# Indexa um documento
result = indexer.index_document(doc_id=1)

print(f"✅ Documento indexado!")
print(f"   Chunks criados: {result['chunks']}")
print(f"   Vetores no Pinecone: {result['vectors_upserted']}")

indexer.close()
```

## 4. Buscar Documentos

```python
from src.indexer import DocumentIndexer

indexer = DocumentIndexer()

# Busca semântica
results = indexer.search(
    query="Como funciona a autenticação?",
    top_k=5
)

# Exibe resultados
for i, result in enumerate(results, 1):
    print(f"\n{i}. Score: {result['score']:.3f}")
    print(f"   Tópico: {result['metadata']['topic']}")
    print(f"   Resumo: {result['metadata']['contextual_summary']}")
    print(f"   Texto: {result['metadata']['text'][:150]}...")

indexer.close()
```

## 5. Usando via CLI

```bash
# Indexar todos os documentos pendentes
python scripts/run_indexer.py --all

# Indexar um documento específico
python scripts/run_indexer.py --doc-id 123

# Buscar
python scripts/run_indexer.py --search "autenticação de usuários"

# Ver estatísticas
python scripts/run_indexer.py --stats
```

## 6. Configurações Importantes

### .env Mínimo

```ini
# Oracle
ORACLE_USER=seu_usuario
ORACLE_PASSWORD=sua_senha
ORACLE_DSN=localhost:1521/XEPDB1

# Pinecone
PINECONE_API_KEY=sua_api_key_pinecone
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=ubchat-docs

# OpenAI
OPENAI_API_KEY=sua_api_key_openai
```

## 7. Testando o Sistema

### Verificar Conexões

```python
from src.indexer import DocumentIndexer

indexer = DocumentIndexer()

# Testa Oracle
docs = indexer.oracle_client.fetch_documents(limit=1)
print(f"✅ Oracle OK - {len(docs)} documento encontrado")

# Testa Pinecone
stats = indexer.pinecone_client.get_index_stats()
print(f"✅ Pinecone OK - {stats.get('total_vector_count', 0)} vetores")

indexer.close()
```

## 8. Exemplos Completos

Veja os exemplos em `/examples`:

- `basic_indexing.py` - Indexação básica
- `batch_indexing.py` - Indexação em lote
- `search_example.py` - Busca avançada

Execute:
```bash
cd examples
python basic_indexing.py
```

## 9. Troubleshooting Comum

### "ModuleNotFoundError: No module named 'src'"
```bash
# Adicione o diretório ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### "Connection refused" (Oracle)
- Verifique se o Oracle está rodando
- Confirme o DSN no .env
- Teste a conexão com sqlplus

### "Invalid API key" (OpenAI/Pinecone)
- Confirme as API keys no .env
- Verifique se não há espaços extras
- Renove as keys se necessário

### Indexação muito lenta
- Desative LLM temporariamente: `use_llm_context=False`
- Reduza o batch_size
- Verifique sua conexão de internet

## 10. Próximos Passos

1. Leia o [README.md](README.md) completo
2. Estude a [ARCHITECTURE.md](ARCHITECTURE.md)
3. Experimente diferentes estratégias de chunking
4. Ajuste os templates de contexto LLM
5. Implemente em produção

## Dúvidas?

- Veja os exemplos em `/examples`
- Consulte a documentação completa
- Verifique os logs em `indexer.log`

---

**Pronto!** Você está pronto para usar o sistema de indexação com contexto enriquecido por LLM! 🚀
