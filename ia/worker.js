// worker.js - O Trabalhador Otimizado (Performance Extrema)
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0");

// Tenta usar WASM se disponível (muito mais rápido que CPU para Conv2D)
async function setupBackend() {
  try {
    // Se quiser suporte real a WASM, precisaria do script tfjs-backend-wasm
    // Por enquanto, otimizamos o modo CPU ao máximo e garantimos que ele seja síncrono
    await tf.setBackend("cpu");
  } catch (e) {
    console.error("Erro ao configurar backend:", e);
  }
}
setupBackend();

class WorkerMinesweeper {
  constructor(rows, cols, mines) {
    this.rows = rows || 9;
    this.cols = cols || 9;
    this.mines = mines || 10;
    this.numCells = rows * cols;
    this.board = new Array(this.numCells);
    this.revealedCount = 0;
    this.flaggedCount = 0;
    this.stateBuffer = new Float32Array(this.numCells * 3); // Buffer persistente
    this.init();
  }

  init() {
    for (let i = 0; i < this.numCells; i++) {
      this.board[i] = {
        mine: false,
        revealed: false,
        flagged: false,
        count: 0,
      };
    }
    let planted = 0;
    while (planted < this.mines) {
      const idx = Math.floor(Math.random() * this.numCells);
      if (!this.board[idx].mine) {
        this.board[idx].mine = true;
        planted++;
      }
    }
    for (let i = 0; i < this.numCells; i++) {
      if (this.board[i].mine) continue;
      const r = Math.floor(i / this.cols);
      const c = i % this.cols;
      let count = 0;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr,
            nc = c + dc;
          if (nr >= 0 && nr < this.rows && nc >= 0 && nc < this.cols) {
            if (this.board[nr * this.cols + nc].mine) count++;
          }
        }
      }
      this.board[i].count = count;
    }
  }

  getStateTensor() {
    // Implementação de INPUT PADDING (Cérebro Fixo 9x9)
    // Igual ao ia.js para compatibilidade
    const maxRows = 9;
    const maxCols = 9;
    const numCellsMax = maxRows * maxCols;

    // Buffer preenchido com -1 (fora do tabuleiro)
    const buf = new Float32Array(numCellsMax * 3).fill(-1);

    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        // Mapeamento: (r,c) do jogo -> (r,c) do cérebro 9x9
        const iSmall = r * this.cols + c;
        const iBig = r * maxCols + c;

        const cell = this.board[iSmall];
        const base = iBig * 3;

        let value = 0;
        if (cell.revealed) value = (cell.count + 1) / 9;

        buf[base] = value;
        buf[base + 1] = cell.revealed ? 0 : 1;
        buf[base + 2] = cell.flagged ? 1 : 0;
      }
    }
    return tf.tensor3d(buf, [maxRows, maxCols, 3]).expandDims(0);
  }

  step(action) {
    // Action vem do cérebro 9x9, precisa traduzir se necessário
    // Mas no worker step() recebe a ação já decodificada (idx linear)
    // O problema é que a rede retorna índice baseado em 9x9
    // Precisamos ajustar isso no loop principal do worker, não aqui.
    // Aqui assumimos que 'action' é o índice correto para o tabuleiro ATUAL.

    const isFlag = action >= this.numCells;
    const idx = isFlag ? action - this.numCells : action;

    if (idx < 0 || idx >= this.numCells) return { reward: -10, done: false }; // Proteção

    const cell = this.board[idx];

    if (cell.revealed) return { reward: -5, done: false };

    if (isFlag) {
      if (cell.flagged) {
        // Desmarcar
        cell.flagged = false;
        this.flaggedCount--;
        return { reward: cell.mine ? -60 : 20, done: false };
      } else {
        // Marcar
        cell.flagged = true;
        this.flaggedCount++;
        return { reward: cell.mine ? 50 : -30, done: false };
      }
    }

    if (cell.flagged) return { reward: -5, done: false };
    if (cell.mine) return { reward: -100, done: true }; // Morte

    const before = this.revealedCount;
    this.reveal(idx);
    const revealedThisTurn = this.revealedCount - before;

    if (this.revealedCount === this.numCells - this.mines) {
      const moves = this.revealedCount + this.flaggedCount; // Estimativa
      const efficiency = this.numCells / Math.max(moves, 1);
      return { reward: 2000 + efficiency * 100, done: true, win: true };
    }

    return { reward: 10 + revealedThisTurn * 5, done: false };
  }

  reveal(idx) {
    const cell = this.board[idx];
    if (cell.revealed || cell.flagged) return;

    cell.revealed = true;
    this.revealedCount++;

    if (cell.count === 0) {
      const r = Math.floor(idx / this.cols);
      const c = Math.floor(idx % this.cols);
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = r + dr,
            nc = c + dc;
          if (nr >= 0 && nr < this.rows && nc >= 0 && nc < this.cols) {
            this.reveal(nr * this.cols + nc);
          }
        }
      }
    }
  }
}

let model = null;

self.onmessage = async function (e) {
  try {
    const { type, payload } = e.data;

    if (type === "INIT_MODEL") {
      if (model) model.dispose();

      const modelTopology =
        typeof payload.topology === "string"
          ? JSON.parse(payload.topology)
          : payload.topology;

      model = await tf.loadLayersModel(
        tf.io.fromMemory({
          modelTopology: modelTopology,
        }),
      );

      model.compile({ optimizer: "adam", loss: "meanSquaredError" });
      self.postMessage({ type: "READY" });
    }

    if (type === "RUN_BATCH") {
      if (!model) {
        throw new Error("Modelo não inicializado no Worker!");
      }

      const { gamesToPlay, elitePopulation, eliteSize, rows, cols, mines } =
        payload;
      const results = [];

      // Estratégia de diversidade populacional
      for (let i = 0; i < gamesToPlay; i++) {
        let weights;

        if (!elitePopulation || elitePopulation.length === 0) {
          // Fallback: usa pesos aleatórios
          weights = model.getWeights().map((w) => ({
            data: w.dataSync(),
            shape: w.shape,
          }));
        } else if (i < elitePopulation.length) {
          // Primeiros N jogos: Elite pura (sem mutação)
          weights = elitePopulation[i];
        } else if (i < elitePopulation.length * 2) {
          // Próximos N jogos: Elite com mutação leve (5%)
          // Usa módulo para garantir que não estoura o índice se elite for pequena
          const parent = elitePopulation[i % elitePopulation.length];
          weights = mutate(parent, 0.05, 0.1);
        } else if (
          i < elitePopulation.length * 4 &&
          elitePopulation.length >= 2
        ) {
          // Próximos 2N jogos: Crossover entre elite (precisa de pelo menos 2 pais)
          const p1 =
            elitePopulation[Math.floor(Math.random() * elitePopulation.length)];
          const p2 =
            elitePopulation[Math.floor(Math.random() * elitePopulation.length)];
          weights = crossover(p1, p2);
          weights = mutate(weights, 0.1, 0.15);
        } else {
          // Resto: Mutação pesada para exploração
          // Se tiver elite (mesmo que 1), usa o primeiro como base
          // Se tiver mais, escolhe um aleatório dos top 50%
          const bestIdx = Math.floor(
            Math.random() * Math.ceil(elitePopulation.length / 2),
          );
          const best = elitePopulation[bestIdx];
          const mutationRate = 0.15 + Math.random() * 0.15; // 15-30%
          weights = mutate(best, mutationRate, 0.2);
        }

        // Converte para tensors e aplica no modelo
        const weightTensors = weights.map((w) => tf.tensor(w.data, w.shape));
        model.setWeights(weightTensors);

        // Joga o jogo
        const game = new WorkerMinesweeper(rows, cols, mines);
        let score = 0;
        let steps = 0;
        const maxSteps = rows * cols * 2;
        let victory = false;

        while (!game.isGameOver && steps < maxSteps) {
          const action = tf.tidy(() => {
            const state = game.getStateTensor();
            const pred = model.predict(state);
            const data = pred.dataSync();

            const board = game.board;
            // Configuração do Cérebro Mestre (9x9) usado no Padding
            const maxCols = 9;
            const numCellsMax = 81; // 9*9

            let bestActionSmall = 0;
            let bestValue = -Infinity;

            // Itera sobre as células do TABULEIRO ATUAL (pequeno)
            for (let r = 0; r < rows; r++) {
              for (let c = 0; c < cols; c++) {
                const iSmall = r * cols + c; // Índice linear no jogo
                const iBig = r * maxCols + c; // Índice linear na rede

                if (!board[iSmall].revealed) {
                  // 1. Avalia CLIQUE (output 0..80 na rede)
                  const valClick = data[iBig];
                  if (valClick > bestValue) {
                    bestValue = valClick;
                    bestActionSmall = iSmall;
                  }

                  // 2. Avalia BANDEIRA (output 81..161 na rede)
                  // O offset de bandeira na rede é numCellsMax (81)
                  const valFlag = data[iBig + numCellsMax];
                  if (valFlag > bestValue) {
                    bestValue = valFlag;
                    // O offset de bandeira no jogo é apenas o tamanho atual e.g. 9
                    bestActionSmall = iSmall + rows * cols;
                  }
                }
              }
            }
            return bestActionSmall;
          });

          const res = game.step(action);
          score += res.reward;
          if (res.done) {
            if (res.win) victory = true;
            break;
          }
          steps++;
        }

        // Salva os pesos apenas dos top performers
        let finalWeights = null;
        if (score > 0 || victory) {
          finalWeights = weightTensors.map((t) => ({
            data: t.dataSync(),
            shape: t.shape,
          }));
        }

        // Limpa tensors
        weightTensors.forEach((t) => t.dispose());

        results.push({ score, victory, weights: finalWeights });
      }

      // Ordena e retorna top N com pesos (N = eliteSize, padrão 10 se não definido)
      results.sort((a, b) => b.score - a.score);

      const topN = eliteSize || 10; // Fallback para 10 se não receber eliteSize

      const responseResults = results.map((r, idx) => ({
        score: r.score,
        victory: r.victory,
        weights: idx < topN ? r.weights : null, // ✅ Top N levam os pesos
      }));

      self.postMessage({ type: "BATCH_DONE", results: responseResults });
    }
  } catch (err) {
    self.postMessage({ type: "ERROR", error: err.message });
  }
};

// Função de crossover (combina dois pais)
function crossover(parent1, parent2) {
  return parent1.map((w, layerIdx) => {
    const p1Data = parent1[layerIdx].data;
    const p2Data = parent2[layerIdx].data;
    const childData = new Float32Array(p1Data.length);

    // Crossover de ponto único ou uniforme
    const crossoverPoint = Math.floor(Math.random() * p1Data.length);
    for (let i = 0; i < p1Data.length; i++) {
      // 50% uniforme, 50% ponto único
      if (Math.random() < 0.5) {
        childData[i] = Math.random() < 0.5 ? p1Data[i] : p2Data[i];
      } else {
        childData[i] = i < crossoverPoint ? p1Data[i] : p2Data[i];
      }
    }

    return { data: childData, shape: w.shape };
  });
}

// Função de mutação
function mutate(weights, rate, amount) {
  return weights.map((w) => {
    const newData = new Float32Array(w.data);
    for (let i = 0; i < newData.length; i++) {
      if (Math.random() < rate) {
        // Mutação gaussiana
        newData[i] += (Math.random() - 0.5) * 2 * amount;
      }
    }
    return { data: newData, shape: w.shape };
  });
}
