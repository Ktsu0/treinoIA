// ia.js - Versão Corrigida e Otimizada com Sistema de Persistência
var aiBrain;
var trainingPaused = false;
var trainingCanceled = false;
var currentCycle = 0;
var totalWins = 0;

class MinesweeperAI {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.gamma = 0.95;
    this.learningRate = 0.0003;
    this.epsilon = 1.0;
    this.epsilonMin = 0.1;
    this.epsilonDecay = 0.998;
    this.memory = [];
    this.maxMemory = 10000;
    this.batchSize = 32;
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [this.rows, this.cols, 3],
        kernelSize: 3,
        filters: 32,
        activation: "relu",
        padding: "same",
      }),
    );
    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: "relu",
        padding: "same",
      }),
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(
      tf.layers.dense({
        units: this.rows * this.cols * 2,
        activation: "linear",
      }),
    );
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError",
    });
    return model;
  }

  getState(board) {
    return tf.tidy(() => {
      const state = board.map((row) =>
        row.map((cell) => {
          let value = 0;
          if (cell.revealed) value = (cell.count + 1) / 9;
          const hidden = cell.revealed ? 0 : 1;
          const flagged = cell.flagged ? 1 : 0;
          return [value, hidden, flagged];
        }),
      );
      return tf.tensor3d(state).expandDims(0);
    });
  }

  async chooseAction(state) {
    const numCells = this.rows * this.cols;
    if (Math.random() < this.epsilon) {
      const flatBoard = board.flat();
      const validActions = [];
      for (let i = 0; i < numCells; i++) {
        if (!flatBoard[i].revealed) {
          validActions.push(i);
          if (!flatBoard[i].flagged) validActions.push(i + numCells);
        }
      }
      return validActions.length > 0
        ? validActions[Math.floor(Math.random() * validActions.length)]
        : Math.floor(Math.random() * (numCells * 2));
    } else {
      return tf.tidy(() => {
        const prediction = this.model.predict(state);
        const predData = prediction.dataSync();
        const flatBoard = board.flat();

        for (let i = 0; i < numCells; i++) {
          if (flatBoard[i].revealed) {
            predData[i] = -Infinity;
            predData[i + numCells] = -Infinity;
          }
          if (flatBoard[i].flagged) {
            predData[i + numCells] = -Infinity;
          }
        }
        return predData.indexOf(Math.max(...predData));
      });
    }
  }

  remember(state, action, reward, nextState, done) {
    this.memory.push({ state, action, reward, nextState, done });
    if (this.memory.length > this.maxMemory) this.memory.shift();
  }

  async replay() {
    if (this.memory.length < this.batchSize) return null;

    // CRÍTICO: Garante que o modelo está compilado antes de treinar
    if (!this.model.optimizer) {
      this.model.compile({
        optimizer: tf.train.adam(this.learningRate),
        loss: "meanSquaredError",
      });
    }

    const batch = [];
    for (let i = 0; i < this.batchSize; i++) {
      batch.push(this.memory[Math.floor(Math.random() * this.memory.length)]);
    }

    const states = tf.concat(batch.map((item) => item.state));
    const nextStates = tf.concat(batch.map((item) => item.nextState));
    const currentQs = this.model.predict(states);
    const nextQs = this.model.predict(nextStates);
    const qArray = await currentQs.array();
    const nextQArray = await nextQs.array();

    for (let i = 0; i < this.batchSize; i++) {
      let target = batch[i].reward;
      if (!batch[i].done) {
        target += this.gamma * Math.max(...nextQArray[i]);
      }
      qArray[i][batch[i].action] = Math.max(-50, Math.min(50, target));
    }

    const h = await this.model.fit(states, tf.tensor2d(qArray), {
      epochs: 1,
      verbose: 0,
    });

    states.dispose();
    nextStates.dispose();
    currentQs.dispose();
    nextQs.dispose();

    return h.history.loss[0];
  }

  decayEpsilon() {
    if (this.epsilon > this.epsilonMin) {
      this.epsilon *= this.epsilonDecay;
    }
  }
}

aiBrain = new MinesweeperAI(9, 9);

// ========== SISTEMA DE EXPORTAÇÃO/IMPORTAÇÃO ==========

async function exportBrain() {
  if (!aiBrain || !aiBrain.model) {
    alert("Nenhum cérebro para exportar!");
    return;
  }

  try {
    // Garante que o modelo está compilado antes de exportar
    if (!aiBrain.model.optimizer) {
      aiBrain.model.compile({
        optimizer: tf.train.adam(aiBrain.learningRate),
        loss: "meanSquaredError",
      });
    }
    const brainData = {
      version: "1.0",
      timestamp: new Date().toISOString(),
      metadata: {
        rows: aiBrain.rows,
        cols: aiBrain.cols,
        epsilon: aiBrain.epsilon,
        currentCycle: currentCycle,
        totalWins: totalWins,
        memorySize: aiBrain.memory.length,
      },
      modelTopology: aiBrain.model.toJSON(null, false),
      weights: aiBrain.model.getWeights().map((w) => ({
        shape: w.shape,
        data: Array.from(w.dataSync()),
      })),
    };

    const jsonString = JSON.stringify(brainData, null, 2);

    // 1. Download para backup do usuário
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `brain-backup-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    // 2. Exibe JSON no console para copiar para brain-data.json
    console.log("✅ Cérebro exportado!");
    console.log("📋 Para atualizar o banco de dados do projeto:");
    console.log("   1. Copie o JSON abaixo");
    console.log("   2. Cole em: ia/brain-data.json");
    console.log("\n" + jsonString);

    alert(
      `✅ Cérebro exportado!\n\n📥 Arquivo baixado\n\n📋 Copie o JSON do console e cole em 'ia/brain-data.json'`,
    );
  } catch (err) {
    console.error("Erro ao exportar:", err);
    alert("Erro ao exportar cérebro!");
  }
}

async function importBrain(file) {
  try {
    const text = await file.text();
    const brainData = JSON.parse(text);

    // Recria o cérebro com as dimensões corretas
    aiBrain = new MinesweeperAI(
      brainData.metadata.rows,
      brainData.metadata.cols,
    );

    // Restaura metadados
    aiBrain.epsilon = brainData.metadata.epsilon;
    currentCycle = brainData.metadata.currentCycle || 0;
    totalWins = brainData.metadata.totalWins || 0;

    // Restaura pesos do modelo
    const weightValues = brainData.weights.map((w) => {
      if (w.shape && w.data) {
        // Novo formato com shape
        return tf.tensor(w.data, w.shape);
      } else {
        // Formato antigo (compatibilidade)
        return tf.tensor(w);
      }
    });
    aiBrain.model.setWeights(weightValues);
    weightValues.forEach((w) => w.dispose());

    // CRÍTICO: Recompila o modelo após carregar os pesos
    aiBrain.model.compile({
      optimizer: tf.train.adam(aiBrain.learningRate),
      loss: "meanSquaredError",
    });

    // Atualiza UI
    updateStats();
    document.getElementById("ia-status").innerText =
      `Cérebro Carregado! (${currentCycle} ciclos)`;

    console.log("✅ Cérebro importado com sucesso!");
    alert(
      `Cérebro carregado!\nCiclos: ${currentCycle}\nVitórias: ${totalWins}`,
    );
  } catch (err) {
    console.error("Erro ao importar:", err);
    alert("Erro ao carregar cérebro! Verifique o arquivo.");
  }
}

// ========== CONTROLE DE TREINO ==========

function togglePause() {
  trainingPaused = !trainingPaused;
  const btn = document.getElementById("pause-btn");
  if (trainingPaused) {
    btn.innerText = "▶️ Retomar";
    btn.style.background = "#27ae60";
    saveBrainToStorage();
  } else {
    btn.innerText = "⏸️ Pausar";
    btn.style.background = "#f39c12";
  }
}

function stopTraining() {
  trainingCanceled = true;
  saveBrainToStorage();
}

async function saveBrainToStorage() {
  try {
    await aiBrain.model.save("localstorage://minesweeper-model");
    localStorage.setItem(
      "minesweeper-metadata",
      JSON.stringify({
        epsilon: aiBrain.epsilon,
        currentCycle: currentCycle,
        totalWins: totalWins,
        timestamp: new Date().toISOString(),
      }),
    );
    console.log("💾 Progresso salvo!");
  } catch (err) {
    console.warn("Erro ao salvar:", err);
  }
}

function updateStats() {
  const cycleEl = document.getElementById("cycle-count");
  const winEl = document.getElementById("win-count");
  const rateEl = document.getElementById("win-rate");

  if (cycleEl) cycleEl.innerText = currentCycle;
  if (winEl) winEl.innerText = totalWins;
  if (rateEl && currentCycle > 0) {
    rateEl.innerText = ((totalWins / currentCycle) * 100).toFixed(1) + "%";
  }
}

// ========== LOOP DE TREINO INFINITO ==========

async function trainIA() {
  const statusEl = document.getElementById("ia-status");
  const pauseBtn = document.getElementById("pause-btn");

  if (statusEl) statusEl.innerText = "Modo: Treinando IA...";
  if (pauseBtn) pauseBtn.style.display = "inline-block";

  trainingMode = true;
  trainingPaused = false;
  trainingCanceled = false;

  const turboCheckbox = document.getElementById("turbo-mode");
  silentMode = turboCheckbox ? turboCheckbox.checked : false;

  if (!rows || !cols) {
    alert("Erro: Inicie um jogo antes de treinar!");
    trainingMode = false;
    if (statusEl) statusEl.innerText = "Modo: Humano";
    return;
  }

  if (!aiBrain || aiBrain.rows !== rows || aiBrain.cols !== cols) {
    aiBrain = new MinesweeperAI(rows, cols);
    // Garante que o modelo está compilado (já é feito no construtor, mas verificamos)
    if (!aiBrain.model.optimizer) {
      aiBrain.model.compile({
        optimizer: tf.train.adam(aiBrain.learningRate),
        loss: "meanSquaredError",
      });
    }
  }

  const progressFill = document.querySelector(".progress-fill");
  const progressContainer = document.querySelector(".training-progress");
  const statsPanel = document.querySelector(".training-stats");

  if (progressContainer) progressContainer.classList.add("active");
  if (statsPanel) statsPanel.classList.add("active");

  const numCells = rows * cols;
  let recentWins = 0;

  // LOOP INFINITO
  while (!trainingCanceled) {
    // Verifica pausa
    while (trainingPaused && !trainingCanceled) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    if (trainingCanceled) break;

    currentCycle++;
    startGame(activeDiff);
    let episodeReward = 0;
    let moves = 0;
    let episodeLoss = 0;
    let lossCount = 0;
    let won = false;

    while (!isGameOver && moves < numCells * 1.5) {
      const state = aiBrain.getState(board);
      const action = await aiBrain.chooseAction(state);
      const isFlag = action >= numCells;
      const cellIdx = isFlag ? action - numCells : action;
      const r = Math.floor(cellIdx / cols);
      const c = cellIdx % cols;

      if (!silentMode) {
        const targetEl = document.getElementById(`cell-${r}-${c}`);
        const currentTargets = document.getElementsByClassName("ia-target");
        while (currentTargets.length > 0)
          currentTargets[0].classList.remove("ia-target");

        if (targetEl) {
          targetEl.classList.add("ia-target");
          await new Promise((res) => setTimeout(res, 15));
        }
      }

      let reward = 0;
      if (isFlag) {
        if (board[r][c].flagged) {
          reward = -30;
        } else {
          const isMine = board[r][c].mine;
          handleRightClick(r, c);
          reward = isMine ? 50 : -50;
        }
      } else {
        const result = handleClick(r, c);
        if (result === "mine") reward = -50;
        else if (result === "safe") reward = 15;
        else if (result === "win") {
          reward = 1000;
          won = true;
        } else reward = -10;
      }

      const nextState = aiBrain.getState(board);
      aiBrain.remember(state, action, reward, nextState, isGameOver);
      episodeReward += Number(reward) || 0;
      moves++;

      const loss = await aiBrain.replay();
      if (loss) {
        episodeLoss += loss;
        lossCount++;
      }

      if (moves % (silentMode ? 50 : 5) === 0) await tf.nextFrame();
    }

    aiBrain.decayEpsilon();

    if (won) {
      totalWins++;
      recentWins++;
    }

    if (currentCycle % 50 === 0) {
      console.log(`✅ Vitórias nos últimos 50 episódios: ${recentWins}/50`);
      recentWins = 0;
      await saveBrainToStorage();
    }

    if (!silentMode || currentCycle % 10 === 0) {
      console.log(
        `[CICLO ${currentCycle}] Rec: ${episodeReward.toFixed(0)} | Loss: ${(episodeLoss / (lossCount || 1)).toFixed(4)} | Eps: ${aiBrain.epsilon.toFixed(3)} | Vitórias: ${totalWins}`,
      );
    }

    updateStats();

    if (statusEl && (!silentMode || currentCycle % 10 === 0))
      statusEl.innerText = `Ciclo ${currentCycle} | Rec: ${episodeReward.toFixed(0)} | Vitórias: ${totalWins}`;
  }

  // Finalização
  await saveBrainToStorage();
  if (statusEl)
    statusEl.innerText = `IA Pausada - ${totalWins} vitórias em ${currentCycle} jogos`;
  if (pauseBtn) pauseBtn.style.display = "none";
  trainingMode = false;
  silentMode = false;
  alert(
    `Treino finalizado!\n\nCiclos: ${currentCycle}\nVitórias: ${totalWins}/${currentCycle} (${((totalWins / currentCycle) * 100).toFixed(1)}%)`,
  );
}

function resetBrain() {
  if (confirm("Resetar cérebro da IA? Todo o aprendizado será perdido.")) {
    const keys = Object.keys(localStorage);
    keys.forEach((key) => {
      if (key.includes("tensorflowjs") || key.includes("minesweeper")) {
        localStorage.removeItem(key);
      }
    });
    currentCycle = 0;
    totalWins = 0;
    updateStats();
    location.reload();
  }
}
