# Nanodegree Engenheiro de Machine Learning
## Projeto final
Cristian Carlos dos Santos 
22 de setembro de 2019

## I. Definição
_(aprox. 1-2 páginas)_

### Visão geral do projeto
Nesta seção, procure fornecer uma visão de alto nível do projeto em linguagem simples. Questões para se perguntar ao escrever esta seção:
- _Foi fornecido um resumo do projeto final, bem como o domínio do problema, a origem do projeto e o conjunto de dados ou entradas?_
- _Foram dadas informações suficientes sobre o contexto para que um leitor desinformado possa entender o domínio e a enunciação problema?_

_(aprox. 1-2 parágrafos)_

#### Resumo

O projeto tem como tema o AWS Deep Racer da Amazon. Como a própria Amazon descreve:

“AWS DeepRacer is a reinforcement learning (RL)-enabled autonomous 1/18th-scale vehicle with supporting services in the AWS Machine Learning ecosystem. It offers an interactive learning system for users of all levels to acquire and refine their skill set in machine learning in general and reinforcement learning in particular. You can use the AWS DeepRacer console to train and evaluate deep reinforcement learning models in simulation and then deploy them to an AWS DeepRacer vehicle for autonomous driving. You can also join AWS DeepRacer League to race in the online Virtual Circuit or the in-person Summit Circuit.” [\[1\]](https://aws.amazon.com/deepracer/?nc1=h_ls)

Esse modelo de competição e aprendizado de RL faz com que aprendamos sobre o assunto que é um campo de estudo fascinante, porém muito complexo. Recentemente a Udacity lançou o AWS DeepRacer Scholarship Challenge e como não havia selecionado a minha proposta para o trabalho final resolvi aceitar esse desafio.

### Descrição do problema
Nesta seção, você irá definir o problema que você está tentando resolver de forma clara, incluindo a estratégia (resumo das tarefas) que você irá utilizar para alcançar a solução desejada. Você deverá também discutir detalhadamente qual será a solução pretendida para este problema. Questões para se perguntar ao escrever esta seção:
- _A enunciação do problema foi claramente definida? O leitor irá entender o que você está esperando resolver?_
- _Você discutiu detalhadamente como irá tentar resolver o problema?_
- _A solução antecipada está claramente definida? O leitor entenderá quais resultados você está procurando?_

O objetivo do meu projeto é desenvolver e fazer com que o carrinho chegue no final completando 100% da pista. Porém, como o custo com o AWS DeepRacer pode se tornar alto rapidamente, procurarei a melhor solução com poucos recursos. Minha meta pessoal é U$50,00.

#### Conjuntos de dados e entradas

O AWS DeepRacer treina os modelos usando o algoritmo PPO (Proximal Policy Optimization). Segundo a vídeo aula “Value Functions” (L4: Reinforcement Learning) do curso AWS DeepRacer [\[2\]](https://www.udacity.com/aws-deepracer-scholarship), esse algoritmo é utilizado por ser eficiente, estável e fácil de usar comparado com outros algoritmos. O Algoritmo usa duas redes neurais durante o treinamento: Policy Network (Actor Network) e Value Network (Critic Network).

-   Policy Network: Decide qual ação tomar de acordo com a imagem recebida no input.
-   Value Network: Estima o resultado cumulativo que provavelmente obteremos, considerando a imagem como uma entrada.

##### Reward

A função de recompensa será o guia do algoritmo. Ela que recompensará as ações positivas, ou seja, o que eu gostaria que o carrinho realizasse. Além disso, ela também penalizará açõs indesejadas, como sair da pista durante a corrida. Para a construção da função de recompensa temos uma entrada de uma variável chamada `params`. Essa variável é uma biblioteca no seguinte formato:

```
{
    "all_wheels_on_track": Boolean,    # flag to indicate if the vehicle is on the track
    "x": float,                        # vehicle's x-coordinate in meters
    "y": float,                        # vehicle's y-coordinate in meters
    "distance_from_center": float,     # distance in meters from the track center 
    "is_left_of_center": Boolean,      # Flag to indicate if the vehicle is on the left side to the track center or not. 
    "heading": float,                  # vehicle's yaw in degrees
    "progress": float,                 # percentage of track completed
    "steps": int,                      # number steps completed
    "speed": float,                    # vehicle's speed in meters per second (m/s)
    "steering_angle": float,          # vehicle's steering angle in degrees
    "track_width": float,              # width of the track
    "waypoints": [[float, float], … ], # list of [x,y] as milestones along the track center
    "closest_waypoints": [int, int]    # indices of the two nearest waypoints.
}

```

Mais informações sobre a função de recompensa estão disponíveis na documentação de desenvolvedores da Amazon Deep Racer [\[3\]](https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-reward-function-input.html).

##### Hyperparâmetros

Os Hiperparâmetros algorítmicos são opções disponíveis nas redes neurais citadas acima. Com eles podemos acelerar ajustes de funções e definir pontos importantes do processo de Reinforcement Learning, como fazer com que nosso algoritmo priorize ações de maior recompensa já descoberta ou explore mais ações a fim de encontrar uma ação melhor em determinado estado. No console da AWS Deep Racer temos  os seguintes parâmetros:
- Gradient descent batch size (Tamanho de lote da descida de gradiente)  
- Number of epochs (Número de epochs)  
- Learning rate (Taxa de aprendizado)  
- Entropy  
- Discount factor (Fator de desconto)  
- Loss type (Tipo de perda)  
- Number of experience episodes between each policy-updating iteration - (Número de episódios de experiência entre cada iteração de atualização de política)

É possível encontrar informações mais detalhadas sobre cada um no guia de desenvolvedores do AWS Deep Racer , na seção de ajustes de parâmetros [\[4\]](https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-console-train-evaluate-models.html#deepracer-iteratively-adjust-hyperparameters).

#### Solução
Para solucionar o problema proposto utilizarei como ponto de partida os exemplos de função de recompensa e hiperparâmetros disponíveis nos exemplos da documentação da AWS Deep Racer. 
##### Reward
Para a função de recompensa utilizarei formas de maximizar ações esperadas, como manter-se na pista, ganhar velocidade e completar voltas. Para isso a ideia é usar como base da função de recompensa os parâmetros `distance_from_center` and `all_wheels_on_track`. Além disso, vou usar como multiplicador da recompensa o `progress` a fim de que a recompensa seja maior de acordo com o avanço na pista.
##### Hyperparâmetros
Para os hiperparâmetros, conforme vídeo “Intro to Tuning Hyperparameters” (L5: Tuning your model - AWS DeepRacer Course): “Figuring out what works best for your model is usually done through trial and error.”


#### Métricas
Nesta seção, você precisará definir claramente as métricas ou cálculos que você irá usar para avaliar o desempenho de um modelo ou resultado no seu projeto. Esses cálculos e métricas devem ser justificadas baseado nas características do problema e domínio do problema. Questões para se perguntar ao escrever esta seção:
- _As métricas que você escolheu para medir o desempenho de seus modelos foram discutidas e definidas de forma clara?_
- _Você forneceu justificativas razoáveis para as métricas escolhidas, baseando-se no problema e solução?_

Para a métrica de avaliação a ideia é usar o **percentual de conclusão da pista**, **o tempo de conclusão de volta** e a visualização das recompensas por iteração. Para facilitar essa análise utilizarei o Jupyter notebook disponibilizado no artigo “Using Jupyter Notebook for analysing DeepRacer’s logs” [\[5\]](https://codelikeamother.uk/using-jupyter-notebook-for-analysing-deepracer-s-logs).



## II. Análise
_(aprox. 2-4 páginas)_

### Exploração dos dados - Treinamento
Nesta seção, é esperado que você analise os dados que você está usando para o problema. Esses dados podem ser tanto na forma de um conjunto de dados (ou conjuntos de dados), dados de entrada (ou arquivos de entrada), ou até um ambiente. O tipo de dados deve ser descrito detalhadamente e, se possível, ter estatísticas e informações básicas apresentadas (tais como discussão dos atributos de entrada ou definição de características das entradas ou do ambiente) Qualquer anormalidade ou qualidade interessante dos dados que possam precisar ser devidamente tratadas devem ser identificadas (tais como características que precisem ser transformadas ou a possibilidade de valores atípicos) Questões para se perguntar ao escrever esta seção:
- _Se exite um conjunto de dados para o problema em questão, você discutiu totalmente as características desse conjunto? Uma amostra dos dados foi oferecida ao leitor?_
- _Se existe um conjunto de dados para o problema, as estatísticas sobre eles foram calculadas e reportadas? Foram discutidos quaisquer resultados relevantes desses cálculos?_
- _Se **não** existe um conjunto de dados para o problema, foi realizada uma discussão sobre o espaço de entrada ou os dados de entrada do problema?_
- _Existem anormalidades ou características acerca do espaço de entrada ou conjunto de dados que necessitem ser direcionados? (variáveis categóricas, valores faltando, valores atípicos, etc.)_

A partir do treino e da avaliação realizada no console do AWS Deep Racer, são gerados arquivos de logs do treinamento e da avaliação. Utilizando o modelo de análise disponível no AWS Deepracer Workshop Lab Github [\[6\]](https://github.com/aws-samples/aws-deepracer-workshops) é possível realizar o download desses logs e iniciar a avaliação, atraveś do arquivo `log-analysis/DeepRacer Log Analysis.ipynb`. Além dos logs de treinamento/avaliação é gerado um modelo referente as redes neurais que fazem parte do algoritmo já explicado acima. No console AWS é possível realizar o download deste modelo para que o mesmo também seja avaliado no Jupyter Notebook já mencionado acima.

Meu objetivo inicial foi encontrar, com a função de recompensa baseada nos padrões básicos dos exemplos da documentação e hyperparâmetros padrões, o menor tempo necessário de treinamento para que o carrinho completasse uma volta. O mínimo de tempo que encontrei na pista **re:Invent 2018** foi 1H. Segue a função de recompensa usada e os hyper parâmetros:

**Função de recompensa:**

    def reward_function(params):
	    # Read input parameters
	    track_width = params['track_width']
	    distance_from_center = params['distance_from_center']
	    all_wheels_on_track = params['all_wheels_on_track']
	    steering = abs(params['steering_angle']) # Only need the absolute steering angle
	    progress = params['progress']
	    speed = params['speed']
	    SPEED_THRESHOLD = 1.0
	    SPEED_THRESHOLD_3 = 3.0
	    # Steering penality threshold, change the number based on your action space setting
	    ABS_STEERING_THRESHOLD = 20

	    # Calculate 3 markers that are at varying distances away from the center line
	    marker_1 = 0.1 * track_width
	    marker_2 = 0.25 * track_width
	    marker_3 = 0.5 * track_width

	    # Give higher reward if the car is closer to center line and vice versa
	    if distance_from_center <= marker_1:
	        reward = 1.0
	    elif distance_from_center <= marker_2:
	        reward = 0.5
	    elif distance_from_center <= marker_3:
	        reward = 0.1
	    else:
	        reward = 1e-3  # likely crashed/ close to off track

	    if not all_wheels_on_track:
	        # Penalize if the car goes off track
	        reward = 1e-3
	    elif speed < SPEED_THRESHOLD:
	        # Penalize if the car goes too slow
	        reward = reward - 0.1
	    else:
	        # High reward if the car stays on track and goes fast
	        reward = reward * speed

	    if steering > ABS_STEERING_THRESHOLD:
	        # Penalize reward if the agent is steering too much
	        reward *= 0.8

	    reward = reward + (reward * (progress / 100))

	    return float(reward)

Essa função de recompensa é busca manter o carrinho na pista, sem grandes zigue-zague e adicionando um multiplicador pela velocidade.

**Hyperparâmetros:**

| Hyperparâmetro              |      Value    |
|-----------------------------|:-------------:|
| Gradient descent batch size | 64            |
|Entropy|0.01|
|Discount factor | 0.999 |
|Loss type|Huber|
|Learning rate|0.0003|
|N. of experience episodes between each policy-updating iteration|20|
|Number of epochs|10|

Os hyperparâmetros serão mantidos estes inicialmente.

##### Estrutura de logs
Ao utilizar o notebook `DeepRacer Log Analysis.ipynb` temos recursos de download os logs direto da AWS, desde que o AWS Client seja configurado. Os arquivos necessário para executar os notebooks avaliados neste relatório estarão todos disponíveis na pasta **log-analysis/** (logs) e **log-analysis/intermediate_checkpoint** (modelos).

##### Notebook de Análise
Inicialmente vamos avaliar o notebook resultante de 1 hora de treinamento com as funções definidas acima. O mesmo encontra-se na pasta `log-analysis/` com nome `06 - DeepRacer Log Analysis - MyTrain 60 min speed.ipynb`.

Os logs de treinamento após serem carregados para um DataFrame são exibidos conforme o exemplo abaixo:
![Exemplo de dados de treinamento AWS Deep Racer](images/Exemplo_Logs.png)

Para avaliar as estatísticas selecionei as colunas que achei mais importantes para a avaliação:
![Exemplo de dados de treinamento AWS Deep Racer](images/Train_Describe.png)

Na imagem é possível avaliar os seguinte pontos:

- **Iteration (2) and episode (39)**: Temos um número baixo de iterações e episódios devido ao baixo tempo de treinamento.
- **Throttle:** Com uma média de 0.7765 é possível avaliar que o carrinho esteve acelerando boa parte do tempo de treinamento.
- **Reward:** Temos uma média de recompensa de 1.05. Se formos avaliar a função de recompensa atual, podemos verificar que essa recompensa é atribuída quando o carrinhos está muito próximo ao centro da pista. Como o foco da corrida é tempo de volta, possivelmente a base da recompensa não poderá ser a distância do centro da pista.
- **Progress:** Bom, chegamos no 100%.

### Visualização exploratória - Treinamento
Nesta seção, você precisará fornecer alguma forma de visualização que sintetize ou evidencie uma característica ou atributo relevante sobre os dados. A visualização deve sustentar adequadamente os dados utilizados. Discuta por que essa visualização foi escolhida e por que é relevante. Questões para se perguntar ao escrever esta seção:
- _Você visualizou uma característica ou um atributo relevante acerca do conjunto de dados ou dados de entrada?_
- _A visualização foi completamente analisada e discutida?_
- _Se um gráfico foi fornecido, os eixos, títulos e dados foram claramente definidos?_

#### Avaliação de recompensa por ações

Para avaliação dos gráficos a seguir é necessário tomar como base que o eixo X e o eixo Y são, respectivamente, colunas X e Y do Dataframe avaliado na seção "notebooks de análise".

##### Avaliação geral

Na imagem abaixo é possível verificar as recompensas retornadas no processo de treinamento.

![Exemplo de dados de treinamento AWS Deep Racer](images/Reward_All_Actions.png)

##### Avaliação iteração 2

Na imagem abaixo é possível verificar as recompensas de uma iteração específica do treinamento.

![Exemplo de dados de treinamento AWS Deep Racer](images/Reward_All_Action_It2.png)

##### Path taken for top reward iterations

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards.png)

Nas imagens acima é possível confirmar o que foi levantado na seção "Notebook de Análise" quanto a questão da recompensa ter uma média próxima a 1 devido a base da composição da recompensa ser a distância do centro da pista. Tanto na imagem de avaliação geral quanto a imagem de avaliação de uma iteração específica, no caso a iteração 2, é visto que a distribuição das recompensas estão muito próximas do centro da pista.

Além disso, o caminho percorrido para as iterações de maior recompensa estão extremamente alinhadas com o centro da pista. 

### Visualização exploratória - Treinamento v1

**Resultado da avaliação pela AWS:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate.png)

Na etapa de avaliação temos que entender como o carrinho se portou na pista durante as três tentativas de completar a corrida.

**Tentativa 1:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_1.png)


**Tentativa 2:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_2.png)


**Tentativa 3:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_3.png)

É possível visualizar que o mesmo padrão do treinamento é repetido durante a fase de avaliação. O Carrinho tentou manter-se no centro da pista, por vezes sacrificando a velocidade para que isso acontecesse.

### Algoritmos e técnicas
Nesta seção, você deverá discutir os algoritmos e técnicas que você pretende utilizar para solucionar o problema. Você deverá justificar o uso de cada algoritmo ou técnica baseado nas características do problema e domínio do problema. Questões para se perguntar ao escrever esta seção:
- _Os algoritmos que serão utilizados, incluindo quaisquer variáveis/parâmetros padrão do projeto, foram claramente definidos?_
- _As técnicas a serem usadas foram adequadamente discutidas e justificadas?_
- _Ficou claro como os dados de entrada ou conjuntos de dados serão controlados pelos algoritmos e técnicas escolhidas?_

Conforme descrito na etapa de **I.Definição** o AWS DeepRacer treina os modelos usando o algoritmo PPO (Proximal Policy Optimization). Esse projeto trata-se de um projeto de otimização do algoritmo já definido e usado pela AWS Deep Racer. Nas etapa de **Conjuntos de dados e entradas** e **Exploração dos Dados** já foram descritos a função de recompensa padrão inicial e os hyperparâmetros usados como ponto de partida para análise.

### Benchmark
Nesta  seção, você deverá definir claramente um resultado de referência (benchmark) ou limiar para comparar entre desempenhos obtidos pela sua solução. O raciocínio por trás da referência (no caso onde não é estabelecido um resultado) deve ser discutido. Questões para se perguntar ao escrever esta seção:
- _Algum resultado ou valor que funcione como referência para a medida de desempenho foi fornecido?_
- _Ficou claro como esse resultado ou valor foi obtido (seja por dados ou por hipóteses)?_

Como referência, me baseio no ganhador do “re:Invent 2018" que completou a volta em 12.68 secs [\[7\]](https://medium.com/vaibhav-malpanis-blog/how-to-win-at-deepracer-league-code-and-model-included-27742b868794). Para este modelo foi usada a seguinte função de recompensa:

```
def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    SPEED_THRESHOLD = 1.0 

    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    if not all_wheels_on_track:
		# Penalize if the car goes off track
        reward = 1e-3
    elif speed < SPEED_THRESHOLD:
		# Penalize if the car goes too slow
        reward = reward + 0.5
    else:
		# High reward if the car stays on track and goes fast
        reward = reward + 1.0

    return float(reward)

```

Já em hyperparâmetros foram utilizados os seguinte:

| Hyperparâmetro              |      Value    |
|-----------------------------|:-------------:|
| Gradient descent batch size | 64            |
|Entropy|0.01|
|Discount factor | 0.666 |
|Loss type|Huber|
|Learning rate|0.0003|
|N. of experience episodes between each policy-updating iteration|20|
|Number of epochs|10|

A partir desses parâmetros como referência desejo trilhar o caminho entre essa ótima referência e os parâmetros padrões. Se possível, aprimorar ainda mais o modelo e testar os resultados. **Porém, é importante salientar que o tempo de treinamento usado pelo competidor não foi informado**.


## III. Metodologia
_(aprox. 3-5 páginas)_

### Pré-processamento de dados
Nesta seção, você deve documentar claramente todos os passos de pré-processamento que você pretende fazer, caso algum seja necessário. A partir da seção anterior, quaisquer anormalidades ou características que você identificou no conjunto de dados deverão ser adequadamente direcionadas e tratadas aqui. Questões para se perguntar ao escrever esta seção:
- _Se os algoritmos escolhidos requerem passos de pré-processamento, como seleção ou transformações de atributos, tais passos foram adequadamente documentados?_
- _Baseado na seção de **Exploração de dados**, se existiram anormalidade ou características que precisem ser tratadas, elas foram adequadamente corrigidas?_
- _Se não é necessário um pré-processamento, foi bem definido o porquê?_

Conforme avaliado na seção de **Exploração de dados - Treinamento** a função de recompensa padrão utilizada apresentou uma forte tendência de manter o carrinho no centro da pista. Isso não é de todo mal, porém tratando-se de uma corrida, a ideia é que o tempo de volta seja melhor.

Diante disso, alterei a função de recompensa para ter como base a velocidade do carrinho. Outra grande mudança é que a distância do centro da pista terá uma flexibilidade maior, apresentando descontos, mesmo que poucos, nas extremidades da pista. Mantive o bônus de recompensa conforme o progresso na pista e a penalização para evitar o zigue-zague.

Assim, a função atual de recompensa ficou a seguinte:

    def reward_function(params):
	    # Read input parameters
	    track_width = params['track_width']
	    distance_from_center = params['distance_from_center']
	    all_wheels_on_track = params['all_wheels_on_track']
	    steering = abs(params['steering_angle'])  # Only need the absolute steering angle
	    progress = params['progress']
	    speed = params['speed']
	    # Steering penality threshold, change the number based on your action space setting
	    ABS_STEERING_THRESHOLD = 20

	    if not all_wheels_on_track:
	        # Penalize if the car goes off track
	        reward = 1e-3
	    else:
	        reward = speed

	    # Calculate 3 markers that are at varying distances away from the center line
	    marker_1 = 0.4 * track_width
	    marker_2 = 0.45 * track_width
	    marker_3 = 0.5 * track_width

	    # Give higher reward if the car is closer to center line and vice versa
	    if distance_from_center <= marker_1:
	        reward *= 1
	    elif distance_from_center <= marker_2:
	        reward *= 0.9
	    elif distance_from_center <= marker_3:
	        reward *= 0.85
	    else:
	        reward = 1e-3  # likely crashed/ close to off track

	    if steering > ABS_STEERING_THRESHOLD:
	        # Penalize reward if the agent is steering too much
	        reward *= 0.8

	    reward = reward + (reward * (progress / 100))


	    return float(reward)


### Implementação
Nesta seção, o processo de escolha de quais métricas, algoritmos e técnicas deveriam ser implementados para os dados apresentados deve estar claramente documentado. Deve estar bastante claro como a implementação foi feita, e uma discussão deve ser elaborada a respeito de quaisquer complicações ocorridas durante o processo.  Questões para se perguntar ao escrever esta seção:
- _Ficou claro como os algoritmos e técnicas foram implementados com os conjuntos de dados e os dados de entrada apresentados?_
- _Houve complicações com as métricas ou técnicas originais que acabaram exigindo mudanças antes de chegar à solução?_
- _Houve qualquer parte do processo de codificação (escrita de funções complicadas, por exemplo) que deveriam ser documentadas?_

Após adequar a função de recompensa conforme especificado na seção **Pré-processamento de dados**, foi realizado um treinamento de 1 hora para avaliar a efetividade desta função de recompensa durante o mesmo tempo de treinamento. Uma observação importante é que não foi realizada nenhuma alteração nos hyperparâmetros, inicialmente. O notebook de análise encontra-se na pasta `log-analysis/` com nome `09 - DeepRacer Log Analysis - Final 60 min.ipynb`.

#### Visualização exploratória - Treinamento v2

Com esse treinamento foram obtidos os seguintes resultados:

##### Avaliação geral de recompensa por ações

Na imagem abaixo é possível verificar as recompensas retornadas no processo de treinamento.

![Exemplo de dados de treinamento AWS Deep Racer](images/Reward_All_Actions_v2.png)

Diferente da primeira função de recompensa, é possível ver que o carrinho está explorando mais a pista. Com isso, é possível verificar mais "erros de percurso" do que na primeira versão.

##### Path taken for top reward iterations

Nessa primeira imagem é possível ver que o carrinho manteve uma certa proximidade com o centro da pista, mas não foi tão rigorosa como a do primeiro modelo.

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v2_1.png)

Gostaria de demonstrar que a segunda iteração mais pontuada teve menos rigor ainda que a primeira em manter o carrinho no centro da pista.

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v2_2.png)


**Resultado da avaliação pela AWS:**

Na avaliação do modelo que tem a base em velocidade, 1 hora de treinamento não é o suficiente para que o mesmo complete 100% da pista.
![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate_v2.png)

É possível ver nas tentativas o uso constante da aceleração, ocasionando a saída da pista.

**Tentativa 1:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_1_v2.png)


**Tentativa 2:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_2_v2.png)


**Tentativa 3:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_3_v2.png)


### Refinamento
Nesta seção, você deverá discutir o processo de aperfeiçoamento dos algoritmos e técnicas usados em sua implementação. Por exemplo, ajuste de parâmetros para que certos modelos obtenham melhores soluções está dentro da categoria de refinamento. Suas soluções inicial e final devem ser registradas, bem como quaisquer outros resultados intermediários significativos, conforme o necessário. Questões para se perguntar ao escrever esta seção:
- _Uma solução inicial foi encontrada e claramente reportada?_
- _O processo de melhoria foi documentado de foma clara, bem como as técnicas utilizadas?_
- _As soluções intermediárias e finais foram reportadas claramente, conforme o processo foi sendo melhorado?_

Como no primeiro modelo (v1) já atingi a meta de concluir 100% da pista, buscarei otimizar o tempo de conclusão de volta. Para isso farei alguns ajustes de hiperparâmetros buscando acelerar o treinamento, lembrando que a meta pessoal é não ultrapassar o custo de U$ 50,00.

Para isso, clonarei a versão 2 do modelo de treinamento, que seria o modelo com função de recompensa focada em velocidade. Com isso aproveitarei o conhecimento já construído na primeira hora de treinamento e adicionarei mais 30 minutos de treinamento com os seguintes ajustes:

- **Learning rate:** 0.0009
- **Gradient descent batch size:** 32

A ideia é que com fluxos de atualizações mais frequentes e saltos maiores, os ajustes do algoritmo sejam acelerados.

Com esses ajustes consegui os seguintes resultados de avaliação na AWS:

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate_v2.png)

Temos um avanço em questão de percentual de progresso da pista, mas não de tempo de volta. Vale ressaltar que 30 minutos é um tempo baixo para uma avaliação mais concreta, mas quero destacar a seguinte iteração do carrinho:

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v3_3.png)

É possível ver que na curva sinalizada o mesmo não priorizou o cnetro da pista e sim a velocidade. É possível ver também abaixo que o mesmo, durante a avaliação saiu da pista exatamente neste ponto:

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_1_v3.png)

Quanto aos gráficos de recompensa por iteração, não foram observados grandes alterações devido a função de recompensa se manter a mesma.

Recapitulando então temos até agora um modelo com uma função de recompensa baseada em velocidade que **foi treinado por 1 hora e meia**. Com isso, adicionarei **mais um treinamento de 60 minutos**, mas ajustando alguns parâmetros de recompensa.

- **Learning rate:** 0.0003
- **Discount factor:** 0.888
- **Gradient descent batch size:** 64

A ideia é caminhar novamente para parâmetros mais conservadores, mas com um pequeno ajuste no fator de desconto, pois acredito que diminuindo o número de passos futuros a serem considerados para recompensa priorize ações momentâneas.

Além disso, avaliando um pouco melhor a função de recompensa, é possível que a variável progress tenha descompensado a recompensa baseada na velocidade. 

Isso pode ter ocorrido devido ao seguinte trecho de código:

`reward = reward + (reward * (progress / 100))`

Como a recompensa é baseada na velocidade podem ocorrer os seguintes casos:

Sendo a velocidade = 1 e o progresso = 25 %, a recompensa seria:
**1.25** = 1 + (1 * (25/100))
	
Sendo a velocidade = 0.8 e o progresso = 75 %, a recompensa seria:
**1.4** = 0.8 + (0.8 * (75/100))

Com isso, é possível que com o aumento do progresso, o modelo prefira diminuir a velocidade para garantir uma recompensa maior.

Para resolver essa situação, resolvi remover esse fator relacionado ao progresso. Outra alteração foi a remoção da penalização baseada na variável `steering_angle` por dois motivos:

-  Em alguns momentos o carrinho reduzia a velocidade em trechos retos da pista somente para poder alinhar-se conforme o angulo definido.
- Como o treinamento inicial do modelo já foi realizado inicialmente com esse parâmetro, os problemas iniciais de zigue-zague form diminuídos. 

![Exemplo de dados de treinamento AWS Deep Racer](images/Slight_Left.png)

Agora queremos focar em velocidade.

## IV. Resultados
_(aprox. 2-3 páginas)_

### Modelo de avaliação e validação
Nesta seção, o modelo final e quaisquer qualidades que o sustentem devem ser avaliadas em detalhe. Deve ficar claro como o modelo final foi obtido e por que tal modelo foi escolhido. Além disso, algum tipo de análise deve ser realizada para validar a robustez do modelo e sua solução, como, por exemplo, manipular os dados de entrada ou o ambiente para ver como a solução do modelo é afetada (técnica chamada de análise sensitiva). Questões para se perguntar ao escrever esta seção:
- _O modelo final é razoável e alinhado com as expectativas de solução? Os parâmetros finais do modelo são apropriados?_
- _O modelo final foi testado com várias entradas para avaliar se o modelo generaliza bem com dados não vistos?_
-_O modelo é robusto o suficiente para o problema? Pequenas perturbações (mudanças) nos dados de treinamento ou no espaço de entrada afetam os resultados de forma considerável?_
- _Os resultados obtidos do modelo são confiáveis?_

**Resultado da avaliação pela AWS:**

Antes de avaliarmos o resultado do progresso x tempo de volta, vamos somente visualizar como o trienamento distribuiu as recompensas por iterações.

![Exemplo de dados de treinamento AWS Deep Racer](images/Reward_All_Actions_v6.png)

Agora vamos verificar como o modelo performou nas 3 tentativas de avaliação da AWS:

**Tentativa 1:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v6_1.png)


**Tentativa 2:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v6_2.png)


**Tentativa 3:**

![Exemplo de dados de treinamento AWS Deep Racer](images/Top_Rewards_v6_3.png)


Agora vamos comprar este modelo com o modelo inicial focado na permanência no centro da pista:

**Modelo padrão (Focado na centralidade da pista):**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate.png)
**Modelo atual (Focado na velocidade):**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate_v6.png)

Do modelo padrão para o modelo desenvolvido encontramos 7,071 segundos de diferença, com o total de 2,5 H de treinamento. 


### Justificativa
Nesta seção, a solução final do seu modelo e os resultados dela obtidos devem ser comparados aos valores de referência (benchmark) que você estabeleceu anteriormente no projeto, usando algum tipo de análise estatística. Você deverá também justificar se esses resultados e a solução são significativas o suficiente para ter resolvido o problema apresentado no projeto. Questões para se perguntar ao escrever esta seção:
- _Os resultados finais encontrados são mais fortes do que a referência reportada anteriormente?_
- _Você analisou e discutiu totalmente a solução final?_
- _A solução final é significativa o suficiente para ter resolvido o problema?_

De certa forma os modelos desenvolvidos apresentaram (Foco Centro da pista e Foco em Velocidade) apresentaram resultados melhores que o modelo de benchmark em até 3 H de treinamento.

**Modelo benchmark:**

É possível verificar na imagem abaixo que a função de recompensa e os hyperparâmetros do modelo benchmark não completaram a volta durante as três tentativas de avaliação do modelo treinado durante 3H.

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluation_Beanchmark_1)

Como o modelo de benchmark não deixa claro o tempo no qual foi treinado, não é possível avaliarmos o resultado de forma mais precisa.

**Modelo padrão (Focado na centralidade da pista):**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate.png)

**Modelo atual (Focado na velocidade):**

![Exemplo de dados de treinamento AWS Deep Racer](images/Evaluate_v6.png)

Quanto a questão "*A solução final é significativa o suficiente para ter resolvido o problema?*":
Avaliando o desafio proposto, completar 100% da pista gastando menos de U$ 50,00 entendo que sim, ela é significativa o suficiente para completar o desafio dando subsídios para uma continuídade do trabalho, inclusive. 

## V. Conclusão
_(aprox. 1-2 páginas)_

### Foma livre de visualização
Nesta seção, você deverá fornecer alguma forma de visualização que enfatize uma qualidade importante do projeto. A visualização é de forma livre, mas deve sustentar de forma razoável um resultado ou característica relevante sobre o problema que você quer discutir. Questões para se perguntar ao escrever esta seção:
- _Você visualizou uma qualidade importante ou relevante acerca do problema, conjunto de dados, dados de entrada, ou resultados?_
- _A visualização foi completamente analisada e discutida?_
- _Se um gráfico foi fornecido, os eixos, títulos e dados foram claramente definidos?_

A principal qualidade que poderia elencar no meu projeto é o conhecimento contruído sobre o funcionamento das funções de recompensas definidas e hyperparâmetros utilizados. 

As visualizações que demonstram de forma mais representativa as diferenças entre a função de recompensa inicial, focada na distância do centro da pista, e a final, focada em velocidade são as seguinte:

**Modelo Padrão Inicial**

![Exemplo de dados de treinamento AWS Deep Racer](images/Forma_Livre_0.png)

**Modelo Final**
![Exemplo de dados de treinamento AWS Deep Racer](images/Forma_Livre.png)

É visível a forma como o modelo padrão inicial não explorava toda extensão da pista, por vezes reduzindo a velocidade do carrinho para isso. Já na segunda imagem é possível verificar como o carrinho por vezes "espalha a curva" para compensar a velocidade. E isso tudo devido a definição base da função recompensa.

As função de recompensas utilizadas utilizam um número limitado de parâmetros. Isso faz com que a possibilidade de parametrizações também seja limitada, porém deixa para o algoritmo de RL a maior parte do aprendizado.

O desenvolvedor da função de recompensa sempre terá que ter um equilibro entre definir uma função de recompensa extremamente complexa e uma abrangente, para que o algoritmo de RL tenha a flexibilidade de encontrar outras formas de "resolver o seu problema" de recompensa.



### Reflexão
Nesta seção, você deverá resumir os procedimentos desde o problema até a solução e discutir um ou dois aspectos  do projeto que você achou particularmente interessante ou difícil. É esperado que você reflita sobre o projeto como um todo de forma a mostrar que você possui um entendimento sólido de todo o processo empregado em seu trabalho. Questões para se perguntar ao escrever esta seção:
- _Você resumiu inteiramente o processo que você utilizou neste projeto?_
- _Houve algum aspecto interessante do projeto?_
- _Houve algum aspecto difícil do projeto?_
- _O modelo e solução final alinham-se com suas expectativas para o problema, e devem ser usadas de forma geral para resolver esses tipos de problemas?_

Basicamente as tarefas de RL são muito exploratórios e abrangentes. Para resolver esse problema escolhido, segmentei da seguinte forma meu trabalho:
- Experimentação acerca do funcionamento da plataforma.
- Definição dos marcos de início, como função padrão de recompensa e modelo de benchmark.
- Avaliação dos dados gerados a partir da primeira rodada de treinamento/avaliação.
- Estudo sobre impactos dos parâmetros da função de recompensa.
- Implementação da nova função de recompensa e ajuste de hyperparâmetros.
- Avaliação dos dados gerados a partir da nova rodada de treinamento/avaliação.
- Estudo sobre possiveis melhorias e pontos de impactos negativos na performance do modelo e função de recompensa.
- Nova rodada de treinamento/avaliação.
- Composição e avaliação dos resultados obtidos.

Como Reinforcement Learning é uma área nova para mim, a complexidade de cenários (ações, estados, recompensas, etc) e o risco de alto custo na plataforma AWS são os maiores desafios do meu projeto.

Quanto a solução final, ela ficou abaixo do resultado inicial que eu esperava (os 11 segundos do modelo de benchmark). Porém, o processo de avaliação foi tão produtivo e mostrou-se muito explicativo que compensou e muito o essa diferença entre o tempo de volta que esperava. Sem falar que mesmo o modelo de benchmark, com treinamento inferior a 3 H não performou tão bem quanto o apresentado pelo seu criador.

### Melhorias
Nesta seção, você deverá discutir como um aspecto da sua implementação poderia ser melhorado. Por exemplo, considere maneiras de tornar a sua implementação mais geral e o que precisaria ser modificado. Você não precisa fazer a melhoria, mas as possíveis soluções que resultariam de tais mudanças devem ser consideradas e comparadas/contrastadas com a sua solução atual. Questões para se perguntar ao escrever esta seção:
- _Existem melhorias futuras que podem ser feitas nos algoritmos ou técnicas que você usou neste projeto?_
- _Existem algoritmos ou técnicas que você pesquisou, porém não soube como implementá-las, mas consideraria usar se você soubesse como?_
- _Se você usou sua solução final como nova referência, você acredita existir uma solução ainda melhor?_

Quanto a melhorias são muitas possibilidades. As principais que eu elencaria seriam as seguintes:

- Estudo do parâmetro `heading` para uso na função de recompensa como um auxiliar na centralização do carrinho na pista.
- Treinamento maior do modelo final gerado para avaliar esse processo com mais iterações para que seja verificado se o modelo ficaria super ajustado.
- Treinamento do modelo em outras pistas, para que o mesmo se adapte as mais variadas possibilidades de pistas.

-----------

**Antes de enviar, pergunte-se. . .**

- _O relatório de projeto que você escreveu segue uma estrutura bem organizada, similar ao modelo do projeto?_
- Cada seção (particularmente **Análise** e **Metodologia**) foi escrita de maneira clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo do seu projeto será capaz de entender suas análises, métodos e resultados?
- Você revisou seu relatório de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
- O código que implementa sua solução está legível e comentado adequadamente?
- O código é executado sem erros e produz resultados similares àqueles reportados?


### Referências:
[1] [https://aws.amazon.com/deepracer/?nc1=h_ls](https://aws.amazon.com/deepracer/?nc1=h_ls)
[2] [https://www.udacity.com/aws-deepracer-scholarship](https://www.udacity.com/aws-deepracer-scholarship)
[3 [https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-reward-function-input.html](https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-reward-function-input.html)
[4] [https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-console-train-evaluate-models.html#deepracer-iteratively-adjust-hyperparameters](https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-console-train-evaluate-models.html#deepracer-iteratively-adjust-hyperparameters)
[5][https://codelikeamother.uk/using-jupyter-notebook-for-analysing-deepracer-s-logs](https://codelikeamother.uk/using-jupyter-notebook-for-analysing-deepracer-s-logs)
[6][https://github.com/aws-samples/aws-deepracer-workshops](https://github.com/aws-samples/aws-deepracer-workshops)
[7][https://medium.com/vaibhav-malpanis-blog/how-to-win-at-deepracer-league-code-and-model-included-27742b868794](https://medium.com/vaibhav-malpanis-blog/how-to-win-at-deepracer-league-code-and-model-included-27742b868794)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI1OTIyOTc5LDEwNjcyNDkyNDksNzc3Mj
EwOTE3LC0xNjc5Nzg3MzQzLC0xMTE5NDA0NDkxLC0xNDY1MjA0
MDQsMTA0OTk5NzE0MSwtODIwNDM0LDQ3NDk5MDg1LDIyODc0OT
A3MSwtMTcyNTE5Mzc5OCwtMjA4NjQ1MjQ3Niw0MjIxMTAxODYs
NzIyODY4OTYsNDI1MDY5MjM4LDE5OTQ2MjM5MzQsLTY5NTY3OD
IxNywyMDk0NDYyOTM0LDExNTU4Njk3MSwtMjUyNTMxOTE0XX0=

-->