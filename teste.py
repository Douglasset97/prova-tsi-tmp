import ollama

# Função para conectar ao Ollama e gerar embeddings
def gerar_embeddings(texto):
    # Conectar ao Ollama (substitua 'SEU_ENDPOINT' e 'SUA_CHAVE' pelas suas credenciais)
    client = ollama.Client(endpoint='https://github.com/Douglasset97/Novo-tmp.git', api_key='AIzaSyD0E-yGfeh8brGCn5JxTrj8jDZaEUNNG2Y')
    
    # Gerar embeddings
    embeddings = client.create_embeddings(texto)
    
    return embeddings

# Função para encontrar palavras similares utilizando Ollama
def encontrar_similares(embeddings, positivo, negativo, topn=5):
    # Exemplo simplificado de manipulação de embeddings
    # Você pode adicionar a lógica para usar vetores positivos e negativos aqui
    # Esta parte do código dependerá da API específica do Ollama e como ela suporta essas operações

    # Exibir os primeiros resultados para fins de exemplo
    print(f"Embeddings gerados: {embeddings[:topn]}")

# Fornecer um texto simples
texto = "Exemplo de frase para gerar embeddings"

# Gerar os vetores de embeddings
vetor_embeddings = gerar_embeddings(texto)

# Realizar a operação com palavras positivas e negativas (adapte conforme necessário)
encontrar_similares(vetor_embeddings, positivo=['woman', 'king'], negativo=['man'], topn=5)
