import ollama
import gensim.downloader as api

# Carregar o modelo de Word2Vec pré-treinado
model = api.load("word2vec-google-news-300")  # Modelo com maior capacidade para capturar relações semânticas complexas

# Realizar a operação correta usando palavras positivas e negativas
resultado = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

# Imprimir resultados
print("Palavras encontradas para a expressão 'Rei - Homem + Mulher':")
for palavra, similaridade in resultado:
    print(f"{palavra}: {similaridade}")
