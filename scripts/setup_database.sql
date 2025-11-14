-- Script para criar a tabela de documentos no Oracle
-- Execute este script antes de usar o indexador

CREATE TABLE documents (
    id NUMBER PRIMARY KEY,
    title VARCHAR2(500),
    content CLOB NOT NULL,
    doc_type VARCHAR2(100),
    source VARCHAR2(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR2(50) DEFAULT 'pending',
    indexed_at TIMESTAMP,
    metadata CLOB,
    CONSTRAINT chk_status CHECK (status IN ('pending', 'indexed', 'error', 'processing'))
);

-- Índices para melhorar performance
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_type ON documents(doc_type);
CREATE INDEX idx_documents_created ON documents(created_at);

-- Sequência para auto-incremento
CREATE SEQUENCE documents_seq START WITH 1 INCREMENT BY 1;

-- Trigger para auto-incremento do ID
CREATE OR REPLACE TRIGGER documents_bi
BEFORE INSERT ON documents
FOR EACH ROW
BEGIN
    IF :new.id IS NULL THEN
        SELECT documents_seq.NEXTVAL INTO :new.id FROM dual;
    END IF;
END;
/

-- Trigger para atualizar updated_at
CREATE OR REPLACE TRIGGER documents_bu
BEFORE UPDATE ON documents
FOR EACH ROW
BEGIN
    :new.updated_at := CURRENT_TIMESTAMP;
END;
/

-- Dados de exemplo (opcional)
INSERT INTO documents (title, content, doc_type, source) VALUES (
    'Manual do Usuário - Sistema de Autenticação',
    'O sistema de autenticação utiliza tokens JWT para gerenciar sessões de usuários.
     Quando um usuário faz login, suas credenciais são validadas e um token é gerado.
     Este token deve ser incluído no header de todas as requisições subsequentes.
     O token expira após 24 horas e precisa ser renovado.',
    'manual',
    'docs/authentication.md'
);

INSERT INTO documents (title, content, doc_type, source) VALUES (
    'Política de Privacidade',
    'Nossa política de privacidade garante a proteção dos dados pessoais dos usuários.
     Coletamos apenas informações necessárias para o funcionamento do serviço.
     Os dados são criptografados em trânsito e em repouso.
     Nunca compartilhamos informações pessoais com terceiros sem consentimento.',
    'policy',
    'docs/privacy.md'
);

INSERT INTO documents (title, content, doc_type, source) VALUES (
    'Guia de Integração de APIs',
    'Para integrar com APIs externas, você precisa configurar as credenciais no painel admin.
     Todas as integrações suportam autenticação OAuth 2.0 e API Keys.
     A documentação completa de cada endpoint está disponível no Swagger.
     Rate limits são aplicados: 1000 requisições por hora para contas gratuitas.',
    'guide',
    'docs/api-integration.md'
);

COMMIT;

-- Consulta para verificar os dados
SELECT id, title, status, created_at FROM documents;
