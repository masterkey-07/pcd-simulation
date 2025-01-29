# Projeto PCD: Simulação e Análise de Modelos de Difusão de Contaminantes em Água

# Objetivo

Criar uma simulação que modele a difusão de contaminantes em um corpo d'água (como um lago ou rio), aplicando conceitos de paralelismo para acelerar o cálculo e observar o comportamento de poluentes ao longo do tempo. O projeto investigará o impacto de OpenMP, CUDA e MPI no tempo de execução e na precisão do modelo.

# Etapas do Projeto

### 1. Estudo do Modelo de Difusão

- Estudar a Equação de Difusão/Transporte transiente, representada por:

$$\frac{\partial C}{\partial t} = D \cdot \nabla^2 C$$

Onde:

- $C$ é a concentração do contaminante
- $t$ é o tempo
- $D$ é o coeficiente de difusão
- $\nabla^2 C$: representa a taxa de variação de concentração no espaço.

A equação diferencial pode ser aproximada (discretizada) no tempo e no espaço usando diferenças finitas em uma grade bidimensional, onde a discretização no espaço implica que cada célula da grade atualiza seu valor com base nas células vizinhas em cada iteração. O cálculo da grade atualizada deverá se repetir para que sejam feitas várias interações discretas no tempo.

### 2. Configuração do Ambiente e Parâmetros da Simulação

- Configurar uma grade 2D onde cada célula representa a concentração de contaminantes em uma região do corpo d'água.
- Definir o coeficiente de difusão \(D\), as condições de contorno (por exemplo, bordas onde o contaminante não se espalha) e as condições iniciais (como uma área de alta concentração de contaminante).
- Definir uma quantidade fixa de interações no tempo como 1000 iterações.

### 3. Implementação com OpenMP (Simulação Local em CPU)

- Usar OpenMP para paralelizar o cálculo de difusão entre os núcleos da CPU. Cada núcleo processa uma parte da grade, aplicando as regras de difusão às células sob sua responsabilidade.

- **Entrega 1**: demonstrar o código em OpenMP e apresentar avaliação de desempenho com relação à versão sequencial.

### 4. Implementação com CUDA (Simulação em GPU)

- Implementar a simulação em CUDA, onde cada célula da grade é processada por uma thread independente na GPU, utilizando um esquema de diferenças finitas para calcular o laplaciano de \(C\).
- A execução em GPU permite simular uma grade maior e observar o ganho de desempenho com CUDA.

- **Entrega 2**: demonstrar o código em CUDA e apresentar avaliação de desempenho com relação às versões anteriores.

### 5. Distribuição com MPI (Simulação em Larga Escala)

- Dividir a grade em sub-regiões e distribuir o processamento entre várias máquinas usando MPI.
- Cada máquina processa uma seção do corpo d'água e troca informações nas bordas com as máquinas vizinhas para garantir a continuidade da difusão de contaminantes entre as regiões.

- **Entrega 3**: demonstrar o código em MPI hibrido (pode incluir trechos em OpenMP e CUDA) e apresentar avaliação de desempenho com relação às versões anteriores, porém destacando a escalabilidade possível apenas com MPI.

### 6. Artigo científico e Discussão dos Resultados

- Criar gráficos que mostrem a evolução da concentração ao longo do tempo e comparar o tempo de execução entre as implementações.
- Discutir as vantagens e limitações de cada abordagem, observando a escalabilidade, precisão e aplicabilidade em simulações ambientais.
- Demonstrar visualmente os resultados que comprovem a corretude da simulação.

- **Entrega Final**: entregar o resultado final no formato de artigo científico (modelo a ser disponibilizado).

# Ponto de Partida para a Implementação da Equação

Para aproximar a Equação de Difusão, podemos usar a seguinte fórmula de diferenças finitas central:

$$C*{i,j}^{t+1} = C*{i,j}^t + D \cdot \Delta t \left( \frac{C*{i+1,j}^t + C*{i-1,j}^t + C*{i,j+1}^t + C*{i,j-1}^t - 4 \cdot C\_{i,j}^t}{\Delta x^2} \right)$$

No arquivo `src/sample.c` tem uma implementação sequencial simples, o qual calcula a difusão do contaminante em uma grade de 2000x2000 ao longo de 500 ciclos. A concentração inicial está configurada no centro da grade, e o coeficiente de difusão \(D\) pode ser ajustado conforme necessário.

Comando para executar o exemplo:

```bash
make sample
```

# Brainstorming

- Sequencial:

  - Gráfico de Speedup
  - Gráfico de Linha do Tempo: Iterações (X)
  - Gráfico de Linha do Tempo: Gridsize (X)

- OMP & MPI:

  - Gráfico de Speedup
  - Gráfico de Performance

- CUDA:

  - Gráfico de Speedup
  - Boxplot para cada Blocksize

- Conclusão
  - Gráfico de Execuçao
  - Gráfico de Speedup: Raw x O3 x OMP x MPI

# Pseudo Código do MPI

```
MAX = 12
LGRID = 1000

Algoritmo(N) {
    submatrix [(LGRID / MAX) + 2][LGRID]

    for iteration in iterations {

        for j in LGRID {
            submatrix[top_border + 1][j] = equation(submatrix, top_border + 1, j)
            submatrix[bottom_border - 1][j] = equation(submatrix, bottom_border - 1, j)
        }

        send_border(N, submatrix[top_border], submatrix[bottom_border]);

        for i in LGRID / MAX {
            for j in LGRID {
                submatrix[i][j] = equation(submatrix, i, j)
            }
        }

        receive_border(N, submatrix[top_border], submatrix[bottom_border]);

        update_submatrix();
    }
}
```
