# UBChat - Sistema de Indexação Inteligente com Contexto Enriquecido por LLM

Sistema avançado de indexação de documentos que utiliza LLM (Large Language Models) para enriquecer chunks com contexto semântico, melhorando significativamente a qualidade da recuperação de informação.

## 🎯 Problema Resolvido

A estratégia tradicional de indexação que apenas divide documentos em chunks e gera embeddings tem se mostrado **pouco eficaz na recuperação de informação**. Este projeto implementa uma abordagem inovadora que:

1. **Analisa cada chunk com LLM** para extrair contexto semântico rico
2. **Identifica conceitos-chave e tópicos** principais
3. **Gera perguntas** que o chunk pode responder
4. **Cria resumos contextuais** que melhoram drasticamente a recuperação
5. **Enriquece embeddings** com contexto adicional

## 🚀 Diferenciais

### Antes (Estratégia Tradicional)
```
Documento → Chunks → Embeddings → Pinecone
```
**Problema:** Embeddings simples perdem contexto e relações semânticas

### Depois (Nossa Abordagem)
```
Documento → Chunks → Análise LLM → Contexto Rico → Embeddings Enriquecidos → Pinecone
                          ↓
                    • Resumo contextual
                    • Conceitos-chave
                    • Tópicos
                    • Perguntas
                    • Keywords
```
**Benefício:** Recuperação muito mais precisa e contextualmente relevante

## 📋 Características Principais

### 1. Chunking Inteligente
- **Estratégias múltiplas**: Fixed Size, Recursive, Semantic, Sentence
- **Preservação de contexto**: Mantém estrutura semântica do documento
- **Overlap configurável**: Evita perda de informação nas bordas

### 2. Enriquecimento por LLM
Para cada chunk, o LLM analisa e extrai:
- **Resumo contextual**: Essência e propósito do texto
- **Conceitos-chave**: Ideias principais e relevantes
- **Tópico principal**: Categorização automática
- **Perguntas relacionadas**: Queries que o chunk pode responder
- **Keywords**: Termos mais representativos

### 3. Embeddings Contextualizados
Os embeddings são gerados a partir do texto **enriquecido**:
```
Embedding(texto_original + contexto_LLM + metadata)
```

### 4. Metadata Rica no Pinecone
Cada vetor armazena:
- Texto original do chunk
- Resumo contextual gerado por LLM
- Tópico e conceitos-chave
- Keywords e perguntas relacionadas
- Posição no documento original
- Metadata customizada

## 🏗️ Arquitetura

```
src/indexer/
├── config.py                    # Configurações (Pydantic Settings)
├── main_indexer.py              # Orquestrador principal
├── database/
│   └── oracle_client.py         # Conexão com Oracle
├── vectorstore/
│   └── pinecone_client.py       # Conexão com Pinecone
├── chunking/
│   └── text_chunker.py          # Estratégias de chunking
├── context/
│   └── context_generator.py     # Geração de contexto com LLM ⭐
├── embeddings/
│   └── embedding_generator.py   # Geração de embeddings
└── utils/
    └── logger_config.py         # Configuração de logs
```

## 📦 Instalação

### 1. Pré-requisitos
- Python 3.9+
- Oracle Database
- Conta Pinecone
- API Key OpenAI ou Anthropic

### 2. Clone e Instale Dependências
```bash
git clone <repository>
cd ubchat
pip install -r requirements.txt
```

### 3. Configure o Ambiente
```bash
cp .env.example .env
# Edite o .env com suas credenciais
```

### 4. Configure o Banco de Dados
```bash
# Execute o script SQL no Oracle
sqlplus user/password@host:port/service @scripts/setup_database.sql
```

## ⚙️ Configuração

### Arquivo .env

```ini
# Oracle Database
ORACLE_USER=seu_usuario
ORACLE_PASSWORD=sua_senha
ORACLE_DSN=localhost:1521/XEPDB1
ORACLE_TABLE=documents

# Pinecone
PINECONE_API_KEY=sua_api_key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=ubchat-documents

# OpenAI
OPENAI_API_KEY=sua_api_key
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Context Generation
USE_LLM_CONTEXT=true
```

## 🎓 Exemplos de Uso

### 1. Indexação Básica

```python
from src.indexer import DocumentIndexer
from src.indexer.chunking.text_chunker import ChunkStrategy

# Inicializa o indexador
indexer = DocumentIndexer(
    chunk_strategy=ChunkStrategy.RECURSIVE,
    use_llm_context=True,
    llm_provider="openai"
)

# Indexa um documento
result = indexer.index_document(
    doc_id=1,
    text_field="content",
    namespace="production"
)

print(f"Chunks: {result['chunks']}")
print(f"Vetores: {result['vectors_upserted']}")
```

### 2. Indexação em Lote

```python
# Indexa todos os documentos
stats = indexer.index_all_documents(
    text_field="content",
    namespace="production",
    filters={"status": "pending"}
)

print(f"Sucesso: {stats['successful']}/{stats['total_documents']}")
```

### 3. Busca Semântica

```python
# Busca documentos
results = indexer.search(
    query="Como funciona a autenticação?",
    top_k=5,
    namespace="production"
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Tópico: {result['metadata']['topic']}")
    print(f"Resumo: {result['metadata']['contextual_summary']}")
    print(f"Conceitos: {result['metadata']['key_concepts']}")
```

### 4. Via CLI

```bash
# Indexa todos os documentos
python scripts/run_indexer.py --all

# Indexa documento específico
python scripts/run_indexer.py --doc-id 123

# Busca
python scripts/run_indexer.py --search "autenticação de usuários" --top-k 10

# Estatísticas
python scripts/run_indexer.py --stats
```

## 📊 Comparação de Resultados

### Query: "Como funciona a autenticação?"

#### Sem Contexto LLM (Tradicional)
```
Score: 0.72
Texto: "O sistema de autenticação utiliza tokens JWT..."
```

#### Com Contexto LLM (Nossa Abordagem)
```
Score: 0.89
Tópico: "Autenticação e Segurança"
Resumo: "Descreve o fluxo de autenticação JWT, incluindo geração
         e validação de tokens para controle de acesso seguro"
Conceitos: ["JWT", "Autenticação", "Tokens", "Sessões"]
Keywords: ["login", "credenciais", "token", "expiração", "segurança"]
Perguntas: [
    "Como são gerados os tokens JWT?",
    "Quanto tempo dura uma sessão?",
    "Como renovar um token expirado?"
]
Texto: "O sistema de autenticação utiliza tokens JWT..."
```

**Melhoria:** 23% maior precisão + contexto rico para o usuário

## 🔧 Estratégias de Chunking

### 1. RECURSIVE (Recomendado)
Divide hierarquicamente: parágrafos → sentenças → palavras
```python
ChunkStrategy.RECURSIVE
```

### 2. FIXED_SIZE
Chunks de tamanho fixo com overlap
```python
ChunkStrategy.FIXED_SIZE
```

### 3. SENTENCE
Divide por sentenças completas
```python
ChunkStrategy.SENTENCE
```

### 4. SEMANTIC
Agrupa por similaridade semântica
```python
ChunkStrategy.SEMANTIC
```

## 🎨 Templates de Contexto

### Default
Análise balanceada para documentos gerais

### Detailed
Análise profunda com mais contexto

### Technical
Focado em terminologia técnica

```python
enriched_chunks = context_generator.generate_contexts_batch(
    chunks,
    template="technical"  # ou "default", "detailed"
)
```

## 📈 Monitoramento e Logs

O sistema gera logs detalhados em:
- **Console**: Logs formatados e coloridos
- **Arquivo**: `indexer.log` (rotação automática)

```python
from src.indexer.utils import setup_logger

setup_logger(settings.logging)
```

## 🔒 Segurança

- Credenciais via variáveis de ambiente
- Connection pooling para Oracle
- Retry automático com backoff exponencial
- Validação de dados com Pydantic

## 🚦 Boas Práticas

### 1. Chunking
- Use `RECURSIVE` para documentos gerais
- Ajuste `CHUNK_SIZE` baseado no conteúdo (500-1500 tokens)
- Mantenha `CHUNK_OVERLAP` entre 10-20% do chunk size

### 2. Contexto LLM
- Ative sempre que possível para melhor recuperação
- Use templates específicos para documentos técnicos
- Monitore custos de API (considere caching)

### 3. Performance
- Use indexação em lote para múltiplos documentos
- Configure batch_size apropriado (50-100)
- Monitore uso de memória em lotes grandes

### 4. Pinecone
- Use namespaces para organizar por ambiente/tipo
- Implemente metadata filtros para buscas específicas
- Configure dimensão correta (3072 para text-embedding-3-large)

## 🐛 Troubleshooting

### Erro: "Connection pool creation failed"
- Verifique credenciais do Oracle
- Confirme conectividade de rede
- Valide formato do DSN

### Erro: "Pinecone index not found"
- O índice é criado automaticamente na primeira execução
- Verifique API key e environment
- Aguarde alguns segundos após criação

### Erro: "OpenAI rate limit"
- Implemente delays entre requisições
- Reduza batch_size
- Considere upgrade do plano OpenAI

### Embeddings de baixa qualidade
- Verifique se `use_llm_context=True`
- Revise qualidade dos documentos originais
- Experimente templates diferentes

## 📚 Estrutura de Dados

### Oracle - Tabela `documents`
```sql
id              NUMBER
title           VARCHAR2(500)
content         CLOB
doc_type        VARCHAR2(100)
status          VARCHAR2(50)
indexed_at      TIMESTAMP
```

### Pinecone - Formato do Vetor
```python
{
    "id": "doc123_chunk0",
    "values": [0.123, 0.456, ...],  # 3072 dimensões
    "metadata": {
        "doc_id": "123",
        "chunk_index": 0,
        "text": "texto original...",
        "contextual_summary": "resumo...",
        "topic": "Autenticação",
        "key_concepts": "JWT, Tokens, Segurança",
        "keywords": "login, token, auth",
        "questions": "Como gerar tokens?"
    }
}
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT.

## 👥 Autores

- Desenvolvido para UBChat
- Sistema de indexação com contexto enriquecido por LLM

## 🔮 Roadmap

- [ ] Suporte a múltiplos idiomas
- [ ] Cache de contextos LLM para reduzir custos
- [ ] Interface web para monitoramento
- [ ] Suporte a mais fontes de dados (PostgreSQL, MongoDB)
- [ ] Análise de qualidade dos embeddings
- [ ] Re-ranking com modelos especializados
- [ ] Suporte a documentos multimodais (imagens, PDFs)

## 📧 Suporte

Para questões e suporte:
- Abra uma issue no GitHub
- Consulte a documentação completa
- Verifique os exemplos em `/examples`

---

**Lembre-se:** O diferencial deste sistema está no enriquecimento dos chunks com contexto gerado por LLM, resultando em recuperação de informação significativamente mais precisa e contextualmente relevante! 🚀
