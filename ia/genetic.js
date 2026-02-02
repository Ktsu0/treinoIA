// genetic.js - Algoritmo Genético "A Escolinha"
const POPULATION_SIZE = 200;
const MUTATION_RATE = 0.05; // 5% de chance de mudar um peso
const MUTATION_STRENGTH = 0.1; // Intensidade da mudança

class HeadlessMinesweeper {
  constructor(rows, cols, mines) {
    this.rows = rows;
    this.cols = cols;
    this.mines = mines;
    this.board = [];
    this.gameOver = false;
    this.victory = false;
    this.moves = 0;
    this.revealedCount = 0;
    this.init();
  }

  init() {
    // Cria tabuleiro vazio
    this.board = new Array(this.rows).fill(0).map(() =>
      new Array(this.cols).fill(0).map(() => ({
        mine: false,
        revealed: false,
        flagged: false,
        count: 0,
      })),
    );

    // Planta minas (sem garantir primeira jogada segura por simplicidade genética - ou garantimos?)
    // Para ser justo, vamos garantir que a ia não perca no primeiro clique se possível
    // Mas para headless rápido, vamos randomizar tudo
    let planted = 0;
    while (planted < this.mines) {
      const r = Math.floor(Math.random() * this.rows);
      const c = Math.floor(Math.random() * this.cols);
      if (!this.board[r][c].mine) {
        this.board[r][c].mine = true;
        planted++;
      }
    }

    // Calcula números
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        if (this.board[r][c].mine) continue;
        let count = 0;
        for (let i = -1; i <= 1; i++) {
          for (let j = -1; j <= 1; j++) {
            const nr = r + i,
              nc = c + j;
            if (
              nr >= 0 &&
              nr < this.rows &&
              nc >= 0 &&
              nc < this.cols &&
              this.board[nr][nc].mine
            ) {
              count++;
            }
          }
        }
        this.board[r][c].count = count;
      }
    }
  }

  // Retorna estado normalizado para a IA (igual ia.js)
  getState() {
    return tf.tidy(() => {
      const state = this.board.map((row) =>
        row.map((cell) => {
          let value = 0;
          // Normaliza contagem (0-8 viram 0.11-1.0)
          if (cell.revealed) value = (cell.count + 1) / 9;
          const hidden = cell.revealed ? 0 : 1;
          const flagged = cell.flagged ? 1 : 0;
          return [value, hidden, flagged];
        }),
      );
      return tf.tensor3d(state).expandDims(0);
    });
  }

  step(action) {
    // Ação: 0 a N-1 (revelar), N a 2N-1 (flag)
    const numCells = this.rows * this.cols;
    const isFlag = action >= numCells;
    const index = isFlag ? action - numCells : action;
    const r = Math.floor(index / this.cols);
    const c = index % this.cols;

    this.moves++;

    // Se movimento inválido (já revelado)
    if (this.board[r][c].revealed) {
      return { reward: -1, done: false }; // Penalidade leve por jogar no lixo
    }

    if (isFlag) {
      if (this.board[r][c].flagged) {
        this.board[r][c].flagged = false;
        return { reward: -0.5, done: false };
      }
      this.board[r][c].flagged = true;
      // Recompensa se marcou mina corretamente?
      // No jogo real não sabemos, mas para TREINO podemos dar feedback?
      // Vamos manter fiel ao jogo: Flag não dá ponto imediato, mas evita morte.
      return { reward: 0.1, done: false };
    } else {
      // Clique
      if (this.board[r][c].flagged) return { reward: -5, done: false }; // Clicar em flag é burrice

      if (this.board[r][c].mine) {
        this.gameOver = true;
        this.victory = false;
        return { reward: -50, done: true }; // Explodiu
      }

      this.reveal(r, c);

      if (this.checkWin()) {
        this.gameOver = true;
        this.victory = true;
        return { reward: 100, done: true };
      }

      return { reward: 2, done: false }; // Sobreviveu e revelou
    }
  }

  reveal(r, c) {
    if (
      r < 0 ||
      r >= this.rows ||
      c < 0 ||
      c >= this.cols ||
      this.board[r][c].revealed
    )
      return;

    this.board[r][c].revealed = true;
    this.revealedCount++;

    if (this.board[r][c].count === 0) {
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) this.reveal(r + i, c + j);
      }
    }
  }

  checkWin() {
    // Vitória se todas as não-minas forem reveladas
    const totalSafe = this.rows * this.cols - this.mines;
    return this.revealedCount === totalSafe;
  }
}

class GeneticPopulation {
  constructor() {
    this.population = []; // Array de pesos (Tensors ou arrays planos)
    this.generation = 0;
    this.bestWeights = null;
    this.bestScore = -Infinity;
    // Usa o modelo existente em 'aiBrain' como base/template
  }

  // Inicializa pegando o cérebro atual como "Adão/Eva"
  async initPopulation() {
    if (!aiBrain || !aiBrain.model) {
      alert("Inicie o jogo primeiro para criar a estrutura neural!");
      return;
    }

    console.log("🧬 Criando população inicial...");
    const baseWeights = await this.getWeightsFromModel(aiBrain.model);

    this.population = [];
    // Cria clones iniciais com variações
    for (let i = 0; i < POPULATION_SIZE; i++) {
      // O primeiro é clone exato se já tivermos um bom, senão variamos todos
      // Se for geração 0 e user já treinou, mantemos o 0 puro
      if (i === 0 && totalWins > 0) {
        this.population.push(baseWeights);
      } else {
        this.population.push(this.mutate(baseWeights));
      }
    }
    console.log(`🧬 População criada: ${POPULATION_SIZE} clones.`);
  }

  async getWeightsFromModel(model) {
    // Extrai pesos como arrays JS simples para fácil manipulação
    const weights = [];
    const updateLayers = model.getWeights();
    for (const t of updateLayers) {
      weights.push(await t.data()); // Float32Array
    }
    return weights;
  }

  async setWeightsToModel(model, weightsData, originalTensors) {
    // Reconstrói tensores
    const newTensors = weightsData.map((data, i) => {
      return tf.tensor(data, originalTensors[i].shape);
    });
    model.setWeights(newTensors);
    newTensors.forEach((t) => t.dispose());
  }

  mutate(weights) {
    // Recebe array de Float32Arrays e retorna copia mutada
    return weights.map((w) => {
      const newW = new Float32Array(w);
      for (let i = 0; i < newW.length; i++) {
        if (Math.random() < MUTATION_RATE) {
          // Mutação: Soma valor aleatório pequeno
          newW[i] += (Math.random() * 2 - 1) * MUTATION_STRENGTH;
        }
      }
      return newW;
    });
  }

  async evolve() {
    if (this.population.length === 0) await this.initPopulation();

    const scores = [];

    // Configurações do jogo baseadas no atual
    const r = aiBrain.rows;
    const c = aiBrain.cols;
    const m = mines || Math.floor(r * c * 0.15);

    // Referência aos shapes originais para reconstrução
    const originalTensors = aiBrain.model.getWeights();

    // === FASE 1: AVALIAÇÃO (JOGOS SIMULTÂNEOS/SEQUENCIAIS) ===
    // Vamos rodar um por um para não travar a memória de vídeo,
    // mas é "rápido" porque é headless.

    console.log(`🏁 Iniciando Geração ${this.generation + 1}...`);

    // Loop da "Escolinha"
    for (let i = 0; i < POPULATION_SIZE; i++) {
      // 1. Carrega cérebro do clone no modelo principal
      await this.setWeightsToModel(
        aiBrain.model,
        this.population[i],
        originalTensors,
      );

      // 2. Roda o jogo Headless
      const game = new HeadlessMinesweeper(r, c, m);
      let score = 0;
      let steps = 0;
      const maxSteps = r * c;

      while (!game.gameOver && steps < maxSteps) {
        const state = game.getState();

        // Escolhe ação (Determinístico/Greedy para avaliação real)
        const prediction = aiBrain.model.predict(state);
        const predData = await prediction.data();

        // Mascara ações inválidas para ajudar (opcional, mas justo pq a IA visual tem isso)
        const flatBoard = game.board.flat();
        for (let k = 0; k < flatBoard.length; k++) {
          if (flatBoard[k].revealed) {
            predData[k] = -Infinity;
            predData[k + flatBoard.length] = -Infinity;
          }
        }

        const action = predData.indexOf(Math.max(...predData)); // Argmax

        state.dispose();
        prediction.dispose();

        // Executa
        const stepRes = game.step(action);
        score += stepRes.reward;
        steps++;
      }

      // Bônus por vitória
      if (game.victory) score += 500;

      // Penalidade por tempo gasto (speed run!)
      score -= steps * 0.1;

      scores.push({ index: i, score: score, victory: game.victory });

      // Pequeno delay a cada X para não travar UI se quiser ver progresso
      if (i % 10 === 0) await tf.nextFrame();
    }

    // === FASE 2: SELEÇÃO ===
    // Ordena por pontuação (maior primeiro)
    scores.sort((a, b) => b.score - a.score);

    const winner = scores[0];
    const winnerWeights = this.population[winner.index];

    console.log(
      `🏆 Melhor da Geração ${this.generation}: Score ${winner.score.toFixed(1)} (Vitória: ${winner.victory})`,
    );

    // Salva estatísticas globais
    this.bestScore = winner.score;
    this.generation++;

    // === FASE 3: EVOLUÇÃO (CLONAGEM + MUTAÇÃO) ===
    const newPopulation = [];

    // Elitismo: Mantém o melhor INTACTO (para não perder progresso)
    newPopulation.push(winnerWeights); // Clone puro do rei

    // Cria 49 mutantes baseados no rei
    for (let i = 1; i < POPULATION_SIZE; i++) {
      newPopulation.push(this.mutate(winnerWeights));
    }

    this.population = newPopulation;

    // Carrega o melhor cérebro de volta na IA principal para o usuário ver
    await this.setWeightsToModel(aiBrain.model, winnerWeights, originalTensors);

    // Atualiza UI se existir
    const statusEl = document.getElementById("ia-status");
    if (statusEl)
      statusEl.innerText = `Geração ${this.generation}: Melhor Score ${winner.score.toFixed(0)}`;

    return winner;
  }
}

// Global
let geneticManager = null;
let geneticRunning = false;

async function startGeneticTraining() {
  if (geneticRunning) {
    geneticRunning = false;
    document.getElementById("btn-genetic").innerText = "🧬 Iniciar Escolinha";
    return;
  }

  if (!aiBrain) {
    alert("Inicie um jogo primeiro!");
    return;
  }

  geneticRunning = true;
  document.getElementById("btn-genetic").innerText = "🛑 Parar Evolução";

  geneticManager = new GeneticPopulation();
  await geneticManager.initPopulation();

  while (geneticRunning) {
    const winner = await geneticManager.evolve();

    // Atualiza estatísticas na tela
    const cycleEl = document.getElementById("cycle-count");
    if (cycleEl) cycleEl.innerText = `Gen ${geneticManager.generation}`;

    // Se houve vitória, conta
    if (winner.victory) {
      totalWins++; // Variável global do ia.js
      updateStats();
    }

    // A cada 5 gerações, salva no localstorage
    if (geneticManager.generation % 5 === 0) {
      saveBrainToStorage();
    }
  }
}
