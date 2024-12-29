import ollama
import numpy as np

# Configurando a chave de API
ollama.api_key = 'AIzaSyD0E-yGfeh8brGCn5JxTrj8jDZaEUNNG2Y'

# Função para gerar embeddings
def gerar_embeddings(texto):
    response = ollama.embedding(texto)
    embeddings = response['data'][0]['embeddings']
    return np.array(embeddings)

# Texto de exemplo
texto_exemplo = "Olá, mundo!"

# Gerando embeddings
vetor_embeddings = gerar_embeddings(texto_exemplo)

# Exibindo o vetor de embeddings
print("Tamanho do vetor de embeddings:", len(vetor_embeddings))
print("Valores iniciais do vetor de embeddings:", vetor_embeddings[:10])
