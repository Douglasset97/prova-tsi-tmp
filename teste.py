import ollama
import gensim.downloader as api

# Carregar um modelo menor, como "glove-wiki-gigaword-50"
model = api.load("glove-wiki-gigaword-50")

# Realizar a operação correta usando as palavras positivas e negativas
resultado = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

# Imprimir resultados
print("Palavras encontradas para a expressão 'Rei - Homem + Mulher':")
for palavra, similaridade in resultado:
    print(f"{palavra}: {similaridade}")
