// Configurações da IA
const TRAINING_CONFIG = {
  episodes: 200, // Ciclos por vez
  gamma: 0.95, // Fator de desconto
  epsilon: 1.0, // Exploração (começa em 100%)
  epsilonDecay: 0.995, // Redução da exploração
  epsilonMin: 0.01,
  learningRate: 0.001,
};

let epsilon = 1.0;

class MinesweeperAI {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    // Entrada: O tabuleiro flat
    model.add(
      tf.layers.dense({
        units: 128,
        activation: "relu",
        inputShape: [this.rows * this.cols],
      }),
    );
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    // Saída: Probabilidade de recompensa para cada célula
    model.add(
      tf.layers.dense({ units: this.rows * this.cols, activation: "linear" }),
    );

    model.compile({
      optimizer: tf.train.adam(TRAINING_CONFIG.learningRate),
      loss: "meanSquaredError",
    });
    return model;
  }

  // Pega o estado atual do board e transforma em array numérico para a IA
  getState() {
    return board.flat().map((cell) => {
      if (!cell.revealed) return -1;
      return cell.count;
    });
  }

  async trainBatch(episodes = 20) {
    console.log("Iniciando treinamento...");
    for (let i = 0; i < episodes; i++) {
      startGame("facil"); // Reinicia o jogo para o novo ciclo
      let episodeReward = 0;

      while (!isGameOver) {
        const state = this.getState();
        let action;
        let modo;

        if (Math.random() < TRAINING_CONFIG.epsilon) {
          modo = "CHUTE (Exploração)";
          action = Math.floor(Math.random() * (rows * cols));
        } else {
          modo = "CÉREBRO (Explotação)";
          const prediction = this.model.predict(tf.tensor2d([state]));
          const data = await prediction.data();
          action = data.indexOf(Math.max(...data));

          // Log do valor de confiança da IA na jogada
          console.log(
            `IA previu recompensa de ${Math.max(...data).toFixed(2)} para esta casa.`,
          );
        }

        const r = Math.floor(action / cols);
        const c = action % cols;

        // Executa a ação e calcula recompensa
        let reward = 0;
        const cell = board[r][c];

        if (cell.revealed || cell.flagged) {
          reward = -2; // Penalidade por clicar onde já foi clicado
        } else {
          handleClick(r, c); // Clica na célula

          if (isGameOver) {
            const won = !cell.mine;
            reward = won ? 10 + 100 / seconds : -10;
          } else {
            reward = 1; // Casa limpa
          }
        }

        episodeReward += reward;

        // Treino simplificado (Online Learning)
        const nextState = this.getState();
        const target = reward + TRAINING_CONFIG.gamma * 0.5; // Simplificação para JS

        // Ajusta o modelo
        await this.model.fit(
          tf.tensor2d([state]),
          tf.tensor2d([Array(rows * cols).fill(target)]),
          { epochs: 1 },
        );
      }

      // Reduz exploração conforme aprende
      if (TRAINING_CONFIG.epsilon > TRAINING_CONFIG.epsilonMin) {
        TRAINING_CONFIG.epsilon *= TRAINING_CONFIG.epsilonDecay;
      }

      console.log(
        `Ciclo ${i + 1} concluído. Recompensa Total: ${episodeReward}`,
      );
    }
    alert("Ciclos finalizado!");
  }
}

// Instanciar a IA
const ai = new MinesweeperAI(9, 9);
