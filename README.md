# RAG - Disruptive Architectures

Projeto desenvolvido para demonstrar a implementação de um **RAG (Retrieval-Augmented Generation)** utilizando como base de conhecimento o site da disciplina **Disruptive Architectures**.

## Integrantes

- Samantha Faruolo Galdi - RM554794  **Turma:** 2TDSA  

- Agatha Yie Won Yun - RM561507  **Turma:** 2TDSA  

- Mirelly Sousa Alves - RM566299  **Turma:** 2TDSA  

- Gabriely Bonfim Silva - RM566242  **Turma:** 2TDSA  

## Base consultada

O conteúdo utilizado pelo RAG é obtido diretamente do site:

https://arnaldojr.github.io/DisruptiveArchitectures/

## Como funciona

O projeto segue o fluxo:

1. Coleta o conteúdo das páginas do site.
2. Divide os textos em pequenos trechos (**chunks**).
3. Transforma os chunks em **embeddings** utilizando Sentence Transformers.
4. Transforma a pergunta do usuário em um embedding.
5. Busca os chunks semanticamente mais próximos da pergunta.
6. Envia os trechos recuperados como contexto para o **Gemini**.
7. Gera uma resposta baseada no conteúdo recuperado.

Fluxo simplificado:

```text
Site
  ↓
Coleta das páginas
  ↓
Divisão em chunks
  ↓
Embeddings
  ↓
Busca semântica
  ↓
Contexto recuperado
  ↓
Gemini
  ↓
Resposta
```

## Tecnologias utilizadas

- Python
- Google Gemini
- Sentence Transformers
- BeautifulSoup
- Requests

## Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
```

## Executando

Execute o projeto com:

```bash
python rag.py
```

Após o processamento do site, o sistema ficará disponível para receber perguntas pelo terminal.

Exemplo:

```text
Pergunta: Quais laboratórios de GenAI estão listados?
```

Para encerrar:

```text
sair
```

## Objetivo

Demonstrar na prática o funcionamento de um sistema RAG, utilizando recuperação semântica de informações para fornecer ao modelo de linguagem um contexto proveniente de uma base de conhecimento específica.