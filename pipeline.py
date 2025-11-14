# ////////////////////////////////////////////////////////////////////////////////////////
#
# Chat_RAG Pipeline - Indexing Module for RAG Application
#
# Unimed Blumenau
# 
# Criado por Breno Andrado / Robson Hostin
#
# Descrição: Cria o pipeline de carga dos documentos incrementar de documentos para o banco
#            de dados vetorizado do Pinecone para criação do RAG para o Chat.
#            Utiliza bibliotecas do landchain para extração dos textos dos arquivos e a 
#            criação dos chunk para envio ao Pinecone.
#            A rotina irá Buscar os documentos da Função Gestão da Qualidade, ainda não enviados
#            ou que tiveram alteração para atualização.
#
# Estrutura: 1) Importação das bibliotecas;
#            2) Conexões com bando de dados
#            3) Funções de apoio;
#            4) Extração do texto dos documentos, chunks e envio para bando de dados vetorizado
#            5) Carga dos documentos para processamento.
#            6) Cria documento PDF com lista de todos documentos da qualidade
#            7) Faz limpeza da pasta temporaria
#
# ////////////////////////////////////////////////////////////////////////////////////////// 
#
# Para executar o Pipeline rode o comando:
# $ docker start chat_rag_pipeline
# $ docker stop chat_rag_pipeline
#
# $ docker logs -f chat_rag_pipeline
#
# //////////////////////////////////////////////////////////////////////////////////////////
# Modificações:
# 2025-11-13 - Breno Andrado: Encapsulamento da classe RAGPipeline
# -----------------------------------------------------------------------------------------
# 1) BIBLIOTECAS
# -----------------------------------------------------------------------------------------

import os
import re
import subprocess
import hashlib
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import List
from dotenv import load_dotenv
import difflib

#Bibliotecas para banco vetorizado
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException

#Bibliotecas para leitura e chunk dos documentos
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

#Bibliotecadas para conexao com o banco de dados
from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
import cx_Oracle

load_dotenv()

# Caminho absoluto para a pasta Conversao_documentos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONVERSAO_DIR = BASE_DIR + "/conversao_documentos"

ARQUIVO_LOG = os.path.join(BASE_DIR, "log", "log_execucao.txt")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

# PINECONE
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_DSUNIBLU")
INDEX_NAME = os.getenv("PINECONE_INDEX", "vectorstore")
# INDEX_NAME = "vector-store-1k-300" # "vector-store-2k5-300" # 
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # mxbai-embed-large = 1024

#Mapeamento dos caminhos Windows (SMB) para caminho Linux montado
MAPPINGS = {
    r"\\blumenau.unimed\dfs\APPS": "/mnt/APPS",
    r"\\blumenau.unimed\dfs\Qualidade": "/mnt/Qualidade",
    r"R:\Qualidade": "/mnt/Qualidade"
}

# -----------------------------------------------------------------------------------------
# 2) CONEXÕES COM BANDO DE DADOS
# -----------------------------------------------------------------------------------------
try:        
    cx_Oracle.init_oracle_client(os.getenv('ORA_INSTANT_CLIENTE'))
except:
    pass

oracle_url = os.getenv("URL_ORACLE_DB")
engine = create_engine(oracle_url)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Documentos_Enviados(Base):
    __tablename__ = 'DS_RAG_DOCUMENTOS'
    __table_args__ = {'schema': 'DATASCIENCE'}

    cd_documento = Column(String, primary_key=True)
    id_doc_rag = Column(String)
    dt_envio = Column(DateTime, default=datetime.now)
    qt_chunk = Column(Integer)
    ie_status = Column(String)
    ds_erro = Column(Text)

class DBManager:
    """Gerenciador de Conexão com o Banco de Dados Oracle"""

    @staticmethod
    def buscar_documentos_para_processar():
        """Busca documentos que precisam ser processados"""
        
        sql  = """
        SELECT a.cd_documento,
            a.nm_documento,
            tasy.Converte_path_storage_web(a.ds_arquivo)    ds_arquivo,
            Nvl((SELECT Listagg(l2.cd_setor_atendimento, ',')
                    FROM   tasy.qua_doc_lib l2
                    WHERE  l2.nr_seq_doc = a.nr_sequencia), 0) cd_setores_liberados
        FROM   tasy.qua_documento a
            LEFT JOIN (SELECT cd_documento,
                                Max(dt_envio) DT_ENVIO
                        FROM   datascience.ds_rag_documentos
                        GROUP  BY cd_documento) e
                    ON e.cd_documento = a.cd_documento
        WHERE  1 = 1
            -- AND ( a.dt_atualizacao > e.dt_envio OR e.dt_envio IS NULL ) --Buscas somente os Documentos que tiveram alteração no tasy
            AND a.ie_situacao = 'A'
            AND regexp_substr(tasy.Converte_path_storage_web(a.ds_arquivo), '[^.]+$', 1, 1) IN ('pdf', 'docx', 'doc' )
            AND a.cd_estabelecimento NOT IN ( 381 )
        
            -- CONDIÇÕES TESTES
            -- AND a.cd_documento IN ( 'DIR-306', 'DIR-070')
            -- AND a.cd_documento IN ( 'DIR-306', 'DIR-070', 'DIR-246', 'DIR-073','DIR-062' )
            -- AND a.cd_documento in ('DE-014','DE-015','DE-016','DE-021','DIR-075','DIR-140','DIR-177','DIR-260','FOR-382','DIR-271')
            -- AND (e.dt_envio is null or e.dt_envio <= to_date('01/10/2025','DD/MM/YYYY')) --Reenviar todos para remover cabeçalhos
            -- AND rownum <= 10
            -- AND 1=2
        """

        try:
            documents = []
            with engine.connect() as conn:
                result = conn.execute(text(sql))

                for row in result:
                    documents.append({
                        "cd_documento": row.cd_documento,
                        "nm_documento": row.nm_documento,
                        "ds_arquivo": row.ds_arquivo,
                        "cd_setores_liberados": row.cd_setores_liberados
                    })

                    if os.name == 'nt':
                        documents[-1]["ds_arquivo"] = row.ds_arquivo.strip()
                    else:
                        documents[-1]["ds_arquivo"] = converte_path_to_linux(row.ds_arquivo.strip())

            return documents
        except Exception as e:
            gerar_log(f"[ERRO] Falha ao buscar documentos no banco de dados: {e}")
            return []

    @staticmethod
    def buscar_lista_documentos():
        """Busca a lista de documentos ativos"""
        
        sql  = """
        SELECT a.cd_documento ||' - '|| a.nm_documento AS "Nome do Documento",
               t.ds_tipo_doc AS "Tipo do Documento",
               s.ds_setor_atendimento AS "Setor Responsável"
        FROM   tasy.qua_documento a
               LEFT JOIN tasy.qua_tipo_doc t
                      ON t.nr_sequencia = a.nr_seq_tipo
               LEFT JOIN tasy.setor_atendimento s
                      ON s.cd_setor_atendimento = a.cd_setor_atendimento
        WHERE  1 = 1
               AND a.ie_situacao = 'A'
               -- AND 1=2
        """

        try:
            documents = []
            with engine.connect() as conn:
                result = conn.execute(text(sql))

                for row in result:
                    documents.append({
                        "nome_documento": row[0],
                        "tipo_documento": row[1] or 'N/A',
                        "setor_responsavel": row[2] or 'N/A'
                    })

            return documents
        except Exception as e:
            gerar_log(f"[ERRO] Falha ao buscar lista de documentos no banco de dados: {e}")
            return []


def grava_envio_documento(cd_documento: str, id_doc_rag: str = None, qt_chunk: Integer = None, ie_status: str = 'OK', ds_erro: str = None):
    """Grava registro dos documentos processados no banco de dados para monitoramento"""
    session = Session()
    try:
        session.add(Documentos_Enviados(
                        cd_documento=cd_documento,
                        id_doc_rag=id_doc_rag,
                        qt_chunk=qt_chunk,
                        ie_status=ie_status,
                        ds_erro=ds_erro
                        )
                    )
        session.commit()
    except IntegrityError:
        session.rollback()
        gerar_log(f"[ERRO] Falha ao gravar registro no banco de dados")
    session.close()


# -----------------------------------------------------------------------------------------
# 3) FUNÇÕES DE APOIO 
# -----------------------------------------------------------------------------------------
def chunk_id(doc_id: str, i: int) -> str:
    # IDs ASCII, curtos e determinísticos; 512 chars é o limite duro do Pinecone
    return f"{doc_id}-c{str(i).zfill(5)}"[:128]


def convert_doc_to_pdf(input_path: str) -> str:
    """
    Converte um arquivo .doc para .docx usando o LibreOffice em modo headless.
    Retorna o caminho do arquivo convertido.
    """
    
    # Caminho completo do executável soffice no Windows C:\Program Files\LibreOffice\program\soffice.exe
    soffice_path = 'soffice'

    # Cria a pasta caso não exista
    os.makedirs(CONVERSAO_DIR, exist_ok=True)

    # Executa o LibreOffice em modo headless
    subprocess.run([
        soffice_path,
        "--headless",
        "--convert-to", "pdf",
        "--outdir", CONVERSAO_DIR,
        input_path
    ], check=True)

    # Nome base do arquivo convertido
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(CONVERSAO_DIR, base_name + ".pdf")

    return output_path


def gerar_log(log):
    """Grava logs de execução"""
    #Imprime log para depuração
    print(log)

    # Cria a pasta caso não exista
    os.makedirs(os.path.dirname(ARQUIVO_LOG), exist_ok=True)

    # Cria variável para armazenar a data fim de execução
    hora = datetime.now()
    with open(ARQUIVO_LOG, 'a', encoding="utf-8-sig") as arquivo:
        arquivo.write(hora.strftime('%d/%m/%Y %H:%M:%S') + f' -> {log}\n')


def converte_path_to_linux(win_path: str) -> str:
    """
    Converte caminho Windows (SMB) para caminho Linux montado.
    """
    # Normaliza barras
    path = win_path.replace("\\", "/")
    
    for win_prefix, linux_prefix in MAPPINGS.items():
        if path.startswith(win_prefix.replace("\\", "/")):
            return path.replace(win_prefix.replace("\\", "/"), linux_prefix, 1)
    
    return path  # se não achar prefixo, retorna como está

# -----------------------------------------------------------------------------------------
# 3.1) FUNÇÕES PARA TRATAMENTO DO TEXTO
# -----------------------------------------------------------------------------------------

def linhas_similares(l1, l2, cutoff=0.85):
    """Retorna True se duas linhas forem parecidas o suficiente."""
    return difflib.SequenceMatcher(None, l1.strip(), l2.strip()).ratio() >= cutoff


def remover_cabecalho_rodape(pages, max_linhas=15, cutoff=0.85):
    """
    Remove cabeçalhos (da 2ª página em diante) e rodapés (de todas as páginas),
    comparando páginas linha a linha.
    """
    resultado = []
    prev_lines = []
    prim_cab = 0 #Controla a primeira pagina em que identificou cabeçalho ou rodapé

    for i, page in enumerate(pages):
        linhas = page.page_content.splitlines()

        # Caso ainda não tiver identificado nenh
        if prim_cab == 0 and i < len(pages) - 1:
            prev_lines = pages[i+1].page_content.splitlines()

        # --- Detecta cabeçalho (a partir da 2ª página) ---
        if i > 0 and prev_lines:
            cabecalho = []
            for l_atual, l_prev in zip(linhas[:max_linhas], prev_lines[:max_linhas]):
                if linhas_similares(l_atual, l_prev, cutoff):
                    cabecalho.append(l_atual)
                    prim_cab += 1
                else:
                    break
            if cabecalho:
                linhas = linhas[len(cabecalho):]

        # --- Detecta rodapé (em todas as páginas) ---
        if prev_lines:
            rodape = []
            for l_atual, l_prev in zip(reversed(linhas[-max_linhas:]), reversed(prev_lines[-max_linhas:])):
                x=0
                if linhas_similares(l_atual, l_prev, cutoff):
                    rodape.insert(0, l_atual)
                    prim_cab += 1
                else:
                    break
            if rodape:
                linhas = linhas[:-len(rodape)]

        # monta novo Document preservando metadata
        novo_doc = Document(
            page_content="\n".join(linhas).strip(),
            metadata=page.metadata
        )
        resultado.append(novo_doc)

        prev_lines = page.page_content.splitlines()

    return resultado


def ajustar_quebras_linha(pages):
    """
    Recebe as paginas dos documento e ajusta as quebras de linhas para
    quando a linha atual não terminar com pontuação ".!?;:" e o primeiro caracter
    da proxima linha for minusculo, então remove a quebra de linha  .
    """
    docs_ajustados = []

    for page in pages:
        texto = page.page_content
        linhas = texto.splitlines()
        resultado = []

        for i, linha in enumerate(linhas):
            linha = linha.strip()

            if i > 0:  # existe linha anterior
                anterior = resultado[-1]

                # condição: anterior não termina em pontuação forte
                # e linha atual começa com minúscula
                if anterior and not re.search(r'[.!?;:]\s*$', anterior) and linha and linha[0].islower():
                    if anterior.endswith('-'):  # quebra com hífen
                        resultado[-1] = anterior[:-1] + linha
                    else:
                        resultado[-1] = anterior + " " + linha
                    continue

            resultado.append(linha)

        texto_final = "\n".join(resultado)

        # Adiciona como Document mantendo o metadata original
        docs_ajustados.append(Document(page_content=texto_final, metadata=page.metadata))

    return docs_ajustados

# -----------------------------------------------------------------------------------------
# 4) EXTRAÇÃO DO TEXTO DOS DOCUMENTOS, CHUNKS E ENVIO PARA BANCO DE DADOS VETORIZADO
# -----------------------------------------------------------------------------------------
class PineconeStore:
    def __init__(self):
        api_key = PINECONE_API_KEY
        if not api_key:
            raise ValueError("Variável de ambiente PINECONE_API_KEY não configurada")

        self.pc = Pinecone(api_key=api_key)

        # Cria índice se não existir
        if INDEX_NAME not in self.pc.list_indexes().names():
            gerar_log(f"Índice '{INDEX_NAME}' não existe; criando…")
            try:
                self.pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
            except PineconeApiException as e:
                # 409 = already exists (race condition)
                if getattr(e, "status", None) != 409:
                    raise
        else:
            gerar_log(f"Índice '{INDEX_NAME}' já existe.")

        self.index = self.pc.Index(INDEX_NAME)
        self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        
        # Debug: verificar estado inicial do índice
        try:
            stats = self.index.describe_index_stats()
            gerar_log(f"[DEBUG] Estado inicial do índice '{INDEX_NAME}':")
            gerar_log(f"[DEBUG] - Dimensão: {stats.get('dimension', 'N/A')}")
            gerar_log(f"[DEBUG] - Total de vetores: {stats.get('total_vector_count', 0)}")
            gerar_log(f"[DEBUG] - Namespaces: {list(stats.get('namespaces', {}).keys())}")
        except Exception as e:
            gerar_log(f"[DEBUG] Erro ao verificar estado do índice: {e}")

    def _ensure_namespace_exists(self, namespace: str):
        """
        Garante que um namespace existe no índice.
        Se não existir, cria um vetor dummy temporário para inicializar o namespace.
        """
        try:
            stats = self.index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if namespace not in namespaces:
                gerar_log(f"[INFO] Namespace '{namespace}' não existe. Criando...")
                
                # Cria um vetor dummy para inicializar o namespace
                import random
                dummy_vector = [random.random() for _ in range(EMBED_DIM)]
                
                self.index.upsert(
                    vectors=[{
                        "id": f"__init__{namespace}",
                        "values": dummy_vector,
                        "metadata": {"__dummy__": True}
                    }],
                    namespace=namespace
                )
                
                # Remove imediatamente o vetor dummy
                self.index.delete(ids=[f"__init__{namespace}"], namespace=namespace)
                gerar_log(f"[INFO] Namespace '{namespace}' criado com sucesso")
            else:
                gerar_log(f"[DEBUG] Namespace '{namespace}' já existe")
                
        except Exception as e:
            gerar_log(f"[WARN] Erro ao verificar/criar namespace: {e}. Continuando...")


    # -------- Atualização segura (delete + upsert) --------
    def upsert_pdf(
            self, 
            file_path:            str, 
            file_extension:       str, 
            document_id:          str, 
            document_name:        str, 
            cd_setores_liberados: str, 
            namespace:            str  = "default", 
            delete_before:        bool = True, 
            batch_size:           int  = 100, 
            chunk_size:           int  = 2500,
            chunk_overlap:        int  = 100, 
            separators:           list = ['\n\n','\n','.']
            ):
        
        # Garante que o namespace existe
        self._ensure_namespace_exists(namespace)

        # 1) Carregar o arquivo conforme a extensão
        if file_extension == ('pdf'):
            loader = PyPDFLoader(file_path)
        elif file_extension == ('docx'):
            loader = Docx2txtLoader(file_path)
        elif file_extension == ('doc'):
            # Não é possível fazer a leitura dos arquivos .doc no langchain, é necessário converter para PDF
            file_path = convert_doc_to_pdf(file_path)
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ("md", "markdown"):
            loader = TextLoader(file_path, encoding="utf-8")

        pages = loader.load()

        # Rmove cabeçalhos e rodapés das paginas intermediárias
        pages = remover_cabecalho_rodape(pages)
        pages = ajustar_quebras_linha(pages)

        # 1.1) Divide o arquivo
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
        #splitter = RecursiveCharacterTextSplitter(**{"chunk_size": 30, "chunk_overlap": 10 })
        docs = splitter.split_documents(pages)
        #docs = splitter.split_text(pages)

        if not docs:
            grava_envio_documento(document_id, ie_status='WARN', ds_erro='Sem texto extraído de: {file_path}')
            gerar_log(f"[WARN] Sem texto extraído de: {file_path}")
            return

        texts: List[str] = [d.page_content for d in docs]
        doc_id = document_id

        # 2) (Opcional) Apagar vetores antigos desse PDF por filtro
        # if delete_before:
            # metadado 'doc_id' é usado como alvo do filtro
            # self.index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)


        # # 2) (Opcional) Apagar vetores antigos desse PDF por filtro
        # if delete_before:
        #     # metadado 'doc_id' é usado como alvo do filtro
        #     gerar_log(f"[DEBUG] Tentando deletar vetores antigos - namespace: '{namespace}', doc_id: '{doc_id}'")
        #     try:
        #         # Verifica se o namespace existe antes de tentar deletar
        #         stats = self.index.describe_index_stats()
        #         gerar_log(f"[DEBUG] Namespaces existentes: {list(stats.get('namespaces', {}).keys())}")
                
        #         if namespace in stats.get('namespaces', {}):
        #             self.index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)
        #             gerar_log(f"[DEBUG] Vetores deletados com sucesso para doc_id: {doc_id}")
        #         else:
        #             gerar_log(f"[DEBUG] Namespace '{namespace}' não existe ainda. Pulando delete.")
        #     except Exception as e:
        #         gerar_log(f"[DEBUG] Erro ao deletar vetores: {e}")
        #         # Continua mesmo se não conseguir deletar (namespace pode não existir ainda)


        # 3) Embeddings
        embeddings = self.embedder.embed_documents(texts)

        if len(embeddings[0]) != EMBED_DIM:
            raise ValueError(
                f"Dimensão do embedding ({len(embeddings[0])}) != EMBED_DIM ({EMBED_DIM}). "
                f"Confirme o modelo e o dimension do índice."
            )

        # 4) Preparar vetores (IDs ASCII seguros)
        vectors = []
        for i, emb in enumerate(embeddings):
            vid = chunk_id(doc_id, i)
            meta = {
                "doc_id": doc_id,                  # usado para update/delete por filtro
                "source": document_id + " - " + document_name, #os.path.abspath(file_path),
                "setores": [x.strip() for x in cd_setores_liberados.split(",")],
                "page": docs[i].metadata.get("page") or 0,
                "text": texts[i],
            }
            vectors.append({"id": vid, "values": emb, "metadata": meta})

        # 5) Upsert em lotes
        gerar_log(f"[DEBUG] Preparando para fazer upsert de {len(vectors)} vetores no namespace '{namespace}'")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            try:
                self.index.upsert(vectors=batch, namespace=namespace)
                gerar_log(f"[DEBUG] Upsert de vetores {i} a {i + len(batch) - 1} concluído.")
                # gerar_log(f"[DEBUG] Batch {i//batch_size + 1} enviado com sucesso ({len(batch)} vetores)")
            except Exception as e:
                gerar_log(f"[ERRO] Falha ao fazer upsert do batch começando em {i}: {e}")
                raise
            
        grava_envio_documento(document_id, id_doc_rag=doc_id, qt_chunk=len(vectors))
        gerar_log(f"[OK] {len(vectors)} chunks upsertados (namespace='{namespace}', doc_id='{doc_id}').")


class RAGPipeline:
    """
    Executa o pipeline de processamento simples dos documentos.
    
    Esse pipeline é o processamento feito até o momento. Inclui:
    - Remoção de cabeçalhos e rodapés
    - Ajuste de quebras de linha
    
    """

    CHUNK_SIZE = 1800
    CHUNK_OVERLAP = 300

    def __init__(self):
        self.store = PineconeStore()
        self.db_manager = DBManager()
        

    def run(self):
        """Executa o pipeline completo"""
        gerar_log("=== Início da Execução ===")

        documents = self._get_documents_to_process()

        if not documents:
            gerar_log("Nenhum documento para processar. Fim da execução.")
            return
        

        for doc in documents:
            self._process_document(doc)

        self._generate_document_list()
        self._cleanup_temporary_files()

    def _get_documents_to_process(self):
        documents = self.db_manager.buscar_documentos_para_processar()

        if not documents:
            return []
        
        return documents

    def _process_document(self, document):
        """Processa um único documento."""

        if not document:
            return
        try:
            self.store.upsert_pdf(
                document["ds_arquivo"], 
                file_extension = os.path.splitext(document["ds_arquivo"])[1].lstrip(".").lower(), 
                document_id = document["cd_documento"], 
                document_name = document["nm_documento"], 
                cd_setores_liberados = document["cd_setores_liberados"], 
                namespace = "default", 
                delete_before = True,
                chunk_size=self.CHUNK_SIZE,
                chunk_overlap=self.CHUNK_OVERLAP
                )
        except Exception as e:
            grava_envio_documento(
                document["cd_documento"], 
                ie_status='ERRO', 
                ds_erro=f'{document["nm_documento"]}: {e}'
                )
            gerar_log(f"[ERRO] {document['nm_documento']}: {e}")


    def _generate_document_list(self):
        """Gera o documento PDF com a lista de documentos ativos."""
        gerar_log('Gerando Lista de Documentos')

        documents = self.db_manager.buscar_lista_documentos()

        if not documents:
            gerar_log("Nenhum documento encontrado para a lista.")
            return

        # Monta conteúdo Markdown
        lines = ["# Lista de Documentos Ativos\n"]
        for doc in documents:
            lines.append(f"## Documento: {doc['nome_documento']}")
            lines.append(f"- **Tipo do Documento:** {doc['tipo_documento']}")
            lines.append(f"- **Setor Responsável:** {doc['setor_responsavel']}")
            lines.append("\n---\n")

        # Salva em arquivo
        ds_arquivo = CONVERSAO_DIR + "/Lista_Documentos.md"
        with open(ds_arquivo, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Envia Lista de Documentos para banco vetorizado
        try:
            self.store.upsert_pdf(
                ds_arquivo, 
                file_extension = 'md', 
                document_id = 'Lista_Documentos', 
                document_name ='Lista_Documentos', 
                cd_setores_liberados = '0', 
                namespace = "default", 
                delete_before = True)
        except Exception as e:
            grava_envio_documento('Lista_Documentos', ie_status='ERRO', ds_erro=f'{e}')

    def _cleanup_temporary_files(self):
        """Limpa os arquivos temporários gerados durante o processamento."""
        p = Path(CONVERSAO_DIR)
        for arquivo in p.iterdir():
            if arquivo.is_file():
                arquivo.unlink()                

        gerar_log('Fim da Execução')

# -----------------------------------------------------------------------------------------
# 5) CARGA DOS DOCUMENTOS PARA PROCESSAMENTO
# -----------------------------------------------------------------------------------------

def main():
    try:
        pipeline = RAGPipeline()
        pipeline.run()
    except Exception as e:
        gerar_log(f"[ERRO] Falha na execução do pipeline: {e}")




if __name__ == "__main__":
    
    main()