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
    this.rows = rows;
    this.cols = cols;
    this.mines = mines;
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
    const buf = this.stateBuffer;
    for (let i = 0; i < this.numCells; i++) {
      const cell = this.board[i];
      const base = i * 3;
      buf[base] = cell.revealed ? (cell.count + 1) / 9 : 0;
      buf[base + 1] = cell.revealed ? 0 : 1;
      buf[base + 2] = cell.flagged ? 1 : 0;
    }
    return tf.tensor3d(buf, [this.rows, this.cols, 3]).expandDims(0);
  }

  step(action) {
    const isFlag = action >= this.numCells;
    const idx = isFlag ? action - this.numCells : action;
    const cell = this.board[idx];

    if (cell.revealed) return { reward: -10, done: false };

    if (isFlag) {
      if (cell.flagged) {
        cell.flagged = false;
        this.flaggedCount--;
        // CORREÇÃO: Punição pesada por desmarcar mina correta (-60)
        // Recompensa por corrigir erro ao desmarcar não-mina (+15)
        return { reward: cell.mine ? -60 : 15, done: false };
      }
      cell.flagged = true;
      this.flaggedCount++;
      if (cell.mine) return { reward: 50, done: false };
      const flagPenalty = 20 + (this.flaggedCount - 1) * 5;
      return { reward: -flagPenalty, done: false };
    }

    if (cell.flagged) return { reward: -10, done: false };
    if (cell.mine) return { reward: -1000, done: true };

    const before = this.revealedCount;
    this.reveal(idx);
    const revealedThisTurn = this.revealedCount - before;

    if (this.revealedCount === this.numCells - this.mines) {
      const moves = this.revealedCount + this.flaggedCount;
      const efficiency = this.numCells / Math.max(moves, 1);
      return { reward: 2000 + efficiency * 100, done: true, win: true };
    }

    return { reward: 5 + revealedThisTurn * 3, done: false };
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

      // O modelo precisa ser carregado com uma estrutura específica de modelTopology
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

      const { gamesToPlay, weights, rows, cols, mines } = payload;
      const results = [];
      const mutationRate = 0.05;

      for (let i = 0; i < gamesToPlay; i++) {
        const mutatedTensors = weights.map((w) => {
          const newW = new Float32Array(w.data);
          if (i > 0) {
            const len = newW.length;
            const numToMutate = Math.floor(len * mutationRate);
            for (let m = 0; m < numToMutate; m++) {
              const idx = Math.floor(Math.random() * len);
              newW[idx] += (Math.random() - 0.5) * 0.2;
            }
          }
          return tf.tensor(newW, w.shape);
        });

        model.setWeights(mutatedTensors);

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
            const numCells = game.numCells;

            let bestAction = 0;
            let bestValue = -Infinity;

            for (let k = 0; k < numCells; k++) {
              if (!board[k].revealed) {
                if (data[k] > bestValue) {
                  bestValue = data[k];
                  bestAction = k;
                }
                const flagAction = k + numCells;
                if (data[flagAction] > bestValue) {
                  bestValue = data[flagAction];
                  bestAction = flagAction;
                }
              }
            }
            return bestAction;
          });

          const res = game.step(action);
          score += res.reward;
          if (res.done) {
            if (res.win) victory = true;
            break;
          }
          steps++;
        }

        let finalWeights = null;
        if (victory || score > 500) {
          finalWeights = mutatedTensors.map((t) => ({
            data: t.dataSync(),
            shape: t.shape,
          }));
        }

        results.push({ score, victory, weights: finalWeights });
        for (let t of mutatedTensors) t.dispose();
      }

      // OTIMIZAÇÃO CRÍTICA: Retorna apenas o MELHOR resultado do lote com pesos
      // Isso evita trafegar gigabytes de dados desnecessários
      results.sort((a, b) => b.score - a.score);

      const responseResults = results.map((r, idx) => ({
        score: r.score,
        victory: r.victory,
        weights: idx === 0 ? r.weights : null, // Só o melhor leva os pesos
      }));

      self.postMessage({ type: "BATCH_DONE", results: responseResults });
    }
  } catch (err) {
    self.postMessage({ type: "ERROR", error: err.message });
  }
};
