import ollama
import gensim.downloader as api

# Carregar o modelo de Word2Vec 
model = api.load("word2vec-google-news-300")

# Realizar a operação correta usando as palavras positivas e negativas
resultado = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

# Imprimir resultados
print("Palavras encontradas para a expressão 'Rei - Homem + Mulher':")
for palavra, similaridade in resultado:
    print(f"{palavra}: {similaridade}")
