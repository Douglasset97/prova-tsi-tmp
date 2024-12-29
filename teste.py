import ollama
import numpy as np

# Configurando a chave de API
ollama.api_key = 'AIzaSyD0E-yGfeh8brGCn5JxTrj8jDZaEUNNG2Y'

# Especificando o modelo
modelo = 'modelo_de_exemplo'

# Função para gerar embeddings
def gerar_embeddings(texto, modelo):
    response = ollama.embeddings(texto, model=modelo)
    embeddings = response['data'][0]['embeddings']
    return np.array(embeddings)

# Texto de exemplo
texto_exemplo = "Olá, a Todos é um Ótimo Ano Novo!"

# Gerando embeddings
vetor_embeddings = gerar_embeddings(texto_exemplo, modelo=modelo)

# Exibindo o vetor de embeddings
print("Tamanho do vetor de embeddings:", len(vetor_embeddings))
print("Valores iniciais do vetor de embeddings:", vetor_embeddings[:10])
