# Trabalho 2 -- Redes Neurais


## Instalação das dependências

Esta implementação requer a instalação de uma versão recente do interpretador Python 3 e da biblioteca NumPy. Caso necessário, essas dependências podem ser instaladas da seguinte forma:

1. Instalar a ferramenta Pipenv: https://docs.pipenv.org

2. Executar, no diretório raiz deste projeto: `$ pipenv install`


## Verificação da corretude

Cálculo de gradientes por *backpropagation*:

```
$ ./backpropagation.py NETWORK_FILE WEIGHTS_FILE DATASET_FILE
```

Cálculo de gradientes por estimativa numérica:

```
$ ./numeric_verification.py NETWORK_FILE WEIGHTS_FILE DATASET_FILE
```

A saída desses scripts tem o mesmo formato que o arquivo *initial_weights.txt* detalhado na especificação do trabalho:

  * cada linha contém uma matriz correspondente a uma camada da rede neural
  * as camadas são ordenadas da entrada para a saída da rede neural
  * cada linha de uma matriz corresponde a um neurônio da camada
  * os elementos de uma linha de uma matriz são separados por vírgula
  * as linhas de uma matriz são separadas por ponto-e-vírgula


## Validação cruzada

```
$ ./cross_validation.py path/to/dataset.txt
```

Os parâmetros de execução podem ser alterados manualmente, editando as constantes definidas no início do arquivo `cross_validation.py`.


## Tabelação dos resultados

O comando a seguir carrega todos os arquivos `*.json` de resultados de um diretório e gera uma tabela ordenada pela média do erro de validação (`J_cv`) de cada rede neural.

```
$ ./results/sort_results.py path/to/results/dir [--csv]
```

Com a opção `--csv`, os resultados são impressos em formato CSV.


## Execução dos testes unitários

Executar, no diretório raiz do projeto:

```
$ pipenv install --dev  # Para instalar a ferramenta pytest, se necessário
$ pytest
```
