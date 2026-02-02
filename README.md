# Inteligência Artificial

> Um repositório educacional focado em **algoritmos** e **estruturas de dados** fundamentais para resolver problemas de IA.

## 🎯 Motivação

Este projeto foi criado para aprofundar o entendimento sobre como máquinas resolvem problemas.

Este repositório explora:

- **Algoritmos de busca**: BFS, DFS, UCS, IDDFS e busca bidirecional.
- **Representação de problemas**: Como abstrair problemas reais em estruturas genéricas
- **Estruturas de dados**: Grafos e árvores de busca
- **Otimizações**: Heurísticas, análise de custos e estratégias de exploração

## 📚 Conteúdo

```
├── notebooks/              # Notebooks interativos com exemplos e visualizações
│   └── structures/        # Implementação das estruturas base
│       ├── problem.py     # Classe genérica para definir problemas
│       ├── graph.py       # Estrutura de grafo
│       └── algorithms.py  # Algoritmos de busca
│
├── main.py               # Script principal de demonstração
├── pyproject.toml        # Configuração do projeto Python
└── README.md             # Este arquivo
```

## 🔍 Conceitos Chave

### Definição de Problema
Cada problema é modelado como uma classe que herda de `Problem[S, A]` onde:
- **S**: Tipo do estado
- **A**: Tipo da ação

Assim, você define:
- Estado inicial
- Teste de objetivo
- Função de sucessores (ações possíveis e seus custos)

### Representação Genérica
Usando `TypeVar` e `Generic` do Python, implementamos soluções que funcionam com **qualquer tipo de problema**, desde puzzle de 8 peças até planejamento de rotas.

## 🚀 Como Usar

1. **Clone o repositório**:
   ```bash
   git clone <repo-url>
   cd inteligencia-artificial
   ```

2. **Instale as dependências**:
   ```bash
   pip install -e .
   ```

3. **Explore os notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```


## 📖 Estrutura de Aprendizado

Este projeto segue uma progressão didática:

1. **Entender problemas**: Modelagem genérica com `Problem[S, A]`
2. **Representar estruturas**: Grafos e estados
3. **Implementar algoritmos**: Busca em profundidade, largura, custo uniforme, etc.

## 💡 Aplicações Práticas

Os algoritmos aqui implementados são usados em:
- Planejamento e navegação de robôs
- Solucionadores de quebra-cabeças (Sudoku, Cubo de Rubik)
- Busca em redes (social networks, recomendação)
- Otimização de rotas
- Jogos de IA

## 📝 Licença

Este projeto é fornecido como material educacional.
