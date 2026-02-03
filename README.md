# Minesweeper AI Training

Projeto de treinamento de IA para jogar Campo Minado usando TensorFlow.js (Deep Q-Learning) e Algoritmos Genéticos.

## Estrutura

- **ia/index.html**: Interface principal.
- **ia/ia.js**: Implementação do Agente DQN (Deep Q-Network).
- **ia/worker.js**: Web Worker para treinamento paralelo (Genetic Algorithm).
- **ia/genetic.js**: Gerenciador da "Escolinha Genética" (Multi-thread).
- **ia/game.js**: Lógica do jogo (Interface Humana/Visual).

## Como Usar

1. Abra a pasta `ia` em um servidor local (ex: Live Server do VSCode ou `npx serve ia`).
2. Acesse `index.html`.
3. **Modo Humano**: Jogue normalmente para entender as regras.
4. **Treinar IA**: Clique em "Treinar IA" para usar o método DQN (Aprendizado por Reforço).
   - Ative o "Modo Turbo" para acelerar (desliga visualização).
5. **Escolinha Genética**: Clique em "Escolinha Genética" para usar evolução paralela (usa 100% da CPU).

## Requisitos

- Navegador moderno com suporte a ES6 e Web Workers.
- (Opcional) Node.js para rodar servidor local.

## Detalhes Técnicos

- **DQN**: Usa Redes Neurais Convolucionais (CNN) para "ver" o tabuleiro.
- **Genetic**: Cria múltiplos workers que jogam milhares de partidas, clona os melhores pesos e aplica mutação.
- **Persistência**: O cérebro da IA é salvo automaticamente no `LocalStorage` do navegador. Você pode baixar/upload via botões na interface.
