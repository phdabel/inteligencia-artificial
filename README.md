# Inteligência Artificial

> Um repositório educacional focado em **algoritmos** e **estruturas de dados** fundamentais para resolver problemas de IA.

## 🎯 Motivação

Este projeto foi criado para aprofundar o entendimento sobre como máquinas resolvem problemas.

Este repositório explora:

- **Algoritmos de busca cega**: BFS, DFS, UCS, IDDFS e busca bidirecional
- **Algoritmos de busca informada**: Greedy Best-First Search e A*
- **Algoritmos de busca local**: Hill Climbing e Simulated Annealing
- **Representação de problemas**: Como abstrair problemas reais em estruturas genéricas
- **Estruturas de dados**: Grafos e árvores de busca
- **Problemas clássicos**: Coloração de mapas, roteamento, carteiro e puzzle de 8 peças

## 📚 Conteúdo

```
├── notebooks/                        # Notebooks interativos com exemplos e visualizações
│   ├── 00-graphs.ipynb               # Introdução a grafos
│   ├── 01-mapcoloring.ipynb          # Problema de coloração de mapas
│   ├── 02-routing.ipynb              # Problema de roteamento
│   ├── 03-postman.ipynb              # Problema do carteiro
│   ├── 04-eight_puzzle.ipynb         # Puzzle de 8 peças
│   └── structures/                   # Implementação das estruturas base
│       ├── problem.py                # Classe genérica para definir problemas
│       ├── graph.py                  # Estrutura de grafo
│       ├── blind_search.py           # Algoritmos de busca cega
│       ├── heuristic_search.py       # Algoritmos de busca informada (heurística)
│       ├── local_search.py           # Algoritmos de busca local
│       └── problems/                 # Problemas modelados
│           ├── map_coloring.py       # Coloração de mapas (CSP)
│           ├── routing.py            # Roteamento em grafos
│           ├── postman.py            # Problema do carteiro
│           └── eight_puzzle.py       # Puzzle de 8 peças (N-puzzle)
│
├── main.py                           # Script principal de demonstração
├── pyproject.toml                    # Configuração do projeto Python
└── README.md                         # Este arquivo
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

## 🧠 Algoritmos Implementados

### Busca Cega (Não Informada)
| Algoritmo | Descrição | Ótimo? | Completo? |
|-----------|-----------|--------|-----------|
| **BFS** | Busca em largura — explora nível a nível | Sim (custo uniforme) | Sim |
| **DFS** | Busca em profundidade — explora ramos até o fim | Não | Não (grafos infinitos) |
| **UCS** | Busca de custo uniforme — expande o nó de menor custo acumulado | Sim | Sim |
| **IDDFS** | Aprofundamento iterativo — combina BFS e DFS | Sim (custo uniforme) | Sim |
| **Bidirecional** | Busca simultânea do início e do objetivo | Sim (custo uniforme) | Sim |

### Busca Informada (Heurística)
| Algoritmo | Descrição | Ótimo? | Completo? |
|-----------|-----------|--------|-----------|
| **Greedy Best-First** | Expande o nó com menor valor heurístico `h(n)` | Não | Não |
| **A\*** | Expande o nó com menor `f(n) = g(n) + h(n)` | Sim (heurística admissível) | Sim |

### Busca Local
| Algoritmo | Descrição |
|-----------|-----------|
| **Hill Climbing** | Move-se para o melhor vizinho (minimização ou maximização) — pode ficar preso em ótimos locais |
| **Simulated Annealing** | Aceita movimentos piores com probabilidade decrescente ao longo do tempo — escapa de ótimos locais |

## 🧩 Problemas Modelados

| Problema | Descrição | Notebook |
|----------|-----------|----------|
| **Coloração de Mapas** | Atribuir cores a regiões respeitando restrições de adjacência (CSP) | `01-mapcoloring.ipynb` |
| **Roteamento** | Encontrar o menor caminho entre dois nós em um grafo ponderado | `02-routing.ipynb` |
| **Carteiro** | Visitar um conjunto de endereços com custo mínimo (com heurística MST) | `03-postman.ipynb` |
| **Puzzle de 8 Peças** | Deslizar peças para atingir a configuração objetivo (N-puzzle genérico) | `04-eight_puzzle.ipynb` |

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
3. **Busca cega**: BFS, DFS, UCS, IDDFS, busca bidirecional
4. **Busca informada**: Greedy Best-First Search, A*
5. **Busca local**: Hill Climbing, Simulated Annealing
6. **Aplicar a problemas reais**: Coloração de mapas, roteamento, carteiro, puzzle de 8 peças

## 💡 Aplicações Práticas

Os algoritmos aqui implementados são usados em:
- Planejamento e navegação de robôs
- Solucionadores de quebra-cabeças (Sudoku, Cubo de Rubik)
- Busca em redes (social networks, recomendação)
- Otimização de rotas
- Jogos de IA

## 📝 Licença

Este projeto é fornecido como material educacional.
