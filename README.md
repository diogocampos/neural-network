## Instalação das dependências

1. Instalar Pipenv, se já não estiver instalado: https://docs.pipenv.org

2. Executar, no diretório raiz deste projeto: `$ pipenv install --dev`


## Execução dos testes

```
$ pipenv run pytest
```


## Verificação da corretude

Cálculo de gradientes por *backpropagation*:

```
$ pipenv run ./backpropagation.py NETWORK_FILE WEIGHTS_FILE DATASET_FILE
```

Cálculo de gradientes por estimativa numérica:

```
$ pipenv run ./numeric_verification.py NETWORK_FILE WEIGHTS_FILE DATASET_FILE
```

A saída desses scripts tem o mesmo formato que o arquivo *initial_weights.txt* detalhado na especificação do trabalho:

  * cada linha contém uma matriz correspondente a uma camada da rede neural
  * as camadas são ordenadas da entrada para a saída da rede neural
  * cada linha de uma matriz corresponde a um neurônio da camada
  * os elementos de uma linha de uma matriz são separados por vírgula
  * as linhas de uma matriz são separadas por ponto-e-vírgula


## Validação cruzada

```
$ pipenv run ./cross_validation.py path/to/dataset.txt
```

Os parâmetros de execução podem ser alterados manualmente, editando as constantes definidas no início do arquivo `cross_validation.py`.
