import ollama
import numpy as np

# Conectando ao Ollama
ollama_api_key = 'AIzaSyD0E-yGfeh8brGCn5JxTrj8jDZaEUNNG2Y'
client = ollama.Client(api_key=ollama_api_key)

# Função para gerar embeddings
def gerar_embeddings(texto):
    response = client.embedding(texto)
    embeddings = response['data'][0]['embeddings']
    return np.array(embeddings)

# Texto de exemplo
texto_exemplo = "Olá a Todos e um Ótimo Final de Ano!"

# Gerando embeddings
vetor_embeddings = gerar_embeddings(texto_exemplo)

# Exibindo o vetor de embeddings
print("Tamanho do vetor de embeddings:", len(vetor_embeddings))
print("Valores iniciais do vetor de embeddings:", vetor_embeddings[:10])