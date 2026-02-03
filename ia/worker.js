// worker.js - O Trabalhador Otimizado (CPU Bound)
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.15.0");

// Configura Backend para CPU (evita gargalo WebGL em threads)
tf.setBackend("cpu");

class WorkerMinesweeper {
  constructor(rows, cols, mines) {
    this.rows = rows;
    this.cols = cols;
    this.mines = mines;
    this.init();
  }

  init() {
    this.board = new Array(this.rows).fill(0).map(() =>
      new Array(this.cols).fill(0).map(() => ({
        mine: false,
        revealed: false,
        flagged: false,
        count: 0,
      })),
    );
    let planted = 0;
    while (planted < this.mines) {
      const r = Math.floor(Math.random() * this.rows);
      const c = Math.floor(Math.random() * this.cols);
      if (!this.board[r][c].mine) {
        this.board[r][c].mine = true;
        planted++;
      }
    }
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
            )
              count++;
          }
        }
        this.board[r][c].count = count;
      }
    }
  }

  getStateTensor() {
    const state = this.board.map((row) =>
      row.map((cell) => {
        let value = 0;
        if (cell.revealed) value = (cell.count + 1) / 9;
        const hidden = cell.revealed ? 0 : 1;
        const flagged = cell.flagged ? 1 : 0;
        return [value, hidden, flagged];
      }),
    );
    return tf.tensor3d(state).expandDims(0);
  }

  step(action) {
    const numCells = this.rows * this.cols;
    const isFlag = action >= numCells;
    const index = isFlag ? action - numCells : action;
    const r = Math.floor(index / this.cols);
    const c = index % this.cols;

    if (this.board[r][c].revealed) return { reward: -50, done: false };
    if (isFlag) {
      if (this.board[r][c].flagged) {
        this.board[r][c].flagged = false;
        return { reward: -5, done: false };
      }
      this.board[r][c].flagged = true;
      return { reward: 0.5, done: false };
    }
    if (this.board[r][c].flagged) return { reward: -50, done: false };
    if (this.board[r][c].mine) return { reward: -100, done: true };

    this.reveal(r, c);
    const wins =
      this.board.flat().filter((x) => x.revealed).length ===
      this.rows * this.cols - this.mines;
    if (wins) return { reward: 2000, done: true, win: true };

    return { reward: 15, done: false };
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
    if (this.board[r][c].count === 0) {
      for (let i = -1; i <= 1; i++)
        for (let j = -1; j <= 1; j++) this.reveal(r + i, c + j);
    }
  }
}

let model = null;

self.onmessage = async function (e) {
  const { type, payload } = e.data;

  if (type === "INIT_MODEL") {
    if (model) model.dispose();
    // Apenas carrega a topologia (estrutura). Os pesos serão injetados depois.
    model = await tf.loadLayersModel(tf.io.fromMemory(payload.topology));
    model.compile({ optimizer: "adam", loss: "meanSquaredError" });
  }

  if (type === "RUN_BATCH") {
    if (!model) return;
    const { gamesToPlay, weights, rows, cols, mines } = payload;
    const results = [];

    // 1. Cria tensores BASE uma única vez (Economiza CPU)
    const baseTensors = weights.map((w) => tf.tensor(w.data, w.shape));

    // 2. Loop de Jogo (Otimizado para ser 100% Síncrono/CPU-Bound)
    for (let i = 0; i < gamesToPlay; i++) {
      // A. Mutação Ultrarrápida em Memória (Sem TF.js ops)
      const mutatedTensors = weights.map((w, wIdx) => {
        // Clone do array puro
        const newW = new Float32Array(w.data);
        // Mutação
        if (i > 0) {
          // O primeiro clone é puro
          const len = newW.length;
          for (let k = 0; k < len; k++) {
            // Mutação de 5%: Operação matemática simples
            if (Math.random() < 0.05) newW[k] += (Math.random() - 0.5) * 0.2;
          }
        }
        return tf.tensor(newW, w.shape);
      });

      // Aplica pesos ao modelo
      // Como o modelo já está na memória, setWeights é rápido com tensores prontos
      model.setWeights(mutatedTensors);

      // B. Simulação do Jogo
      const game = new WorkerMinesweeper(rows, cols, mines);
      let score = 0;
      let steps = 0;
      const maxSteps = rows * cols * 2;
      let victory = false;

      while (
        !game.board.flat().some((c) => c.mine && c.revealed) &&
        !victory &&
        steps < maxSteps
      ) {
        // OTIMIZAÇÃO CRÍTICA: USAR TIDY + DATASYNC
        // Tidy limpa memória da GPU/CPU imediatamente
        // dataSync força a thread a esperar o resultado (CPU não dorme)
        const action = tf.tidy(() => {
          const state = game.getStateTensor();
          const pred = model.predict(state);
          const data = pred.dataSync(); // <--- O segredo da performance CPU

          const flat = game.board.flat();
          // Mascara (manual é mais rápido que tensor ops aqui)
          for (let k = 0; k < flat.length; k++) {
            if (flat[k].revealed) {
              data[k] = -Infinity;
              data[k + flat.length] = -Infinity;
            }
          }
          return data.indexOf(Math.max(...data));
        });

        const res = game.step(action);
        score += res.reward;
        if (res.done) {
          if (res.win) victory = true;
          break;
        }
        steps++;
      }

      // Recupera pesos SE for vencedor (para não trafegar tudo)
      let finalWeights = null;
      if (victory || score > 500) {
        finalWeights = mutatedTensors.map((t) => ({
          data: t.dataSync(),
          shape: t.shape,
        }));
      }

      results.push({ score, victory, weights: finalWeights });

      // Limpa tensores mutados DESSA rodada
      mutatedTensors.forEach((t) => t.dispose());
    }

    // Limpa tensores BASE
    baseTensors.forEach((t) => t.dispose());

    self.postMessage({ type: "BATCH_DONE", results });
  }
};
