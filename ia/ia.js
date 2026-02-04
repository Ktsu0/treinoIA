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
    this.epsilon = 1.0; // Começa com 100% exploração (para aprender do zero)
    this.epsilonMin = 0.05; // Mínimo de 5% exploração
    this.epsilonDecay = 0.998; // Decay mais lento para explorar mais no início (antes: 0.995)
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
      })
    );
    model.add(
      tf.layers.conv2d({
        kernelSize: 3,
        filters: 64,
        activation: "relu",
        padding: "same",
      })
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(
      tf.layers.dense({
        units: this.rows * this.cols * 2,
        activation: "linear",
      })
    );
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: "meanSquaredError",
    });
    return model;
  }

  getState(board) {
    return tf.tidy(() => {
      const numCells = this.rows * this.cols;
      const buffer = new Float32Array(numCells * 3);
      const flat = board.flat();

      for (let i = 0; i < numCells; i++) {
        const cell = flat[i];
        let value = 0;
        if (cell.revealed) value = (cell.count + 1) / 9;
        const hidden = cell.revealed ? 0 : 1;
        const flagged = cell.flagged ? 1 : 0;

        const base = i * 3;
        buffer[base] = value;
        buffer[base + 1] = hidden;
        buffer[base + 2] = flagged;
      }
      return tf.tensor3d(buffer, [this.rows, this.cols, 3]).expandDims(0);
    });
  }

  async chooseAction(state) {
    const numCells = this.rows * this.cols;
    const flatBoard = board.flat();

    // EXPLORAÇÃO: Escolhe ação aleatória válida
    if (Math.random() < this.epsilon) {
      const validActions = [];
      for (let i = 0; i < numCells; i++) {
        if (!flatBoard[i].revealed) {
          validActions.push(i);
          validActions.push(i + numCells);
        }
      }
      if (validActions.length === 0)
        return Math.floor(Math.random() * (numCells * 2));
      return validActions[Math.floor(Math.random() * validActions.length)];
    }

    // EXPLORAÇÃO: Usa rede neural mas filtra ações inválidas
    return tf.tidy(() => {
      const prediction = this.model.predict(state);
      const predData = prediction.dataSync();

      let bestAction = 0;
      let bestValue = -Infinity;

      // Encontra melhor ação válida em O(N)
      for (let i = 0; i < numCells; i++) {
        if (!flatBoard[i].revealed) {
          // Clique
          if (predData[i] > bestValue) {
            bestValue = predData[i];
            bestAction = i;
          }
          // Bandeira
          const flagIdx = i + numCells;
          if (predData[flagIdx] > bestValue) {
            bestValue = predData[flagIdx];
            bestAction = flagIdx;
          }
        }
      }

      return bestAction;
    });
  }

  remember(state, action, reward, nextState, done) {
    this.memory.push({ state, action, reward, nextState, done });
    if (this.memory.length > this.maxMemory) {
      const old = this.memory.shift();
      // Limpar memória da GPU para não vazar (Importante!)
      if (old.state) old.state.dispose();
      if (old.nextState) old.nextState.dispose();
    }
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
      qArray[i][batch[i].action] = Math.max(-1000, Math.min(2500, target));
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
      `✅ Cérebro exportado!\n\n📥 Arquivo baixado\n\n📋 Copie o JSON do console e cole em 'ia/brain-data.json'`
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
      brainData.metadata.cols
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
    document.getElementById(
      "ia-status"
    ).innerText = `Cérebro Carregado! (${currentCycle} ciclos)`;

    console.log("✅ Cérebro importado com sucesso!");
    alert(
      `Cérebro carregado!\nCiclos: ${currentCycle}\nVitórias: ${totalWins}`
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
      })
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

// ia.js - Versão Otimizada para Performance Extrema
async function trainIA() {
  const statusEl = document.getElementById("ia-status");
  const pauseBtn = document.getElementById("pause-btn");

  if (statusEl) statusEl.innerText = "Modo: Treinando IA...";
  if (pauseBtn) pauseBtn.style.display = "inline-block";

  trainingMode = true;
  trainingPaused = false;
  trainingCanceled = false;

  const turboCheckbox = document.getElementById("turbo-mode");

  // Captura o estado inicial do turbo
  silentMode = turboCheckbox ? turboCheckbox.checked : false;

  if (!rows || !cols) {
    alert("Erro: Inicie um jogo antes de treinar!");
    trainingMode = false;
    if (statusEl) statusEl.innerText = "Modo: Humano";
    return;
  }

  if (!aiBrain || aiBrain.rows !== rows || aiBrain.cols !== cols) {
    aiBrain = new MinesweeperAI(rows, cols);
    // Garante compilação
    if (!aiBrain.model.optimizer) {
      aiBrain.model.compile({
        optimizer: tf.train.adam(aiBrain.learningRate),
        loss: "meanSquaredError",
      });
    }
  }

  const progressContainer = document.querySelector(".training-progress");
  const statsPanel = document.querySelector(".training-stats");

  if (progressContainer) progressContainer.classList.add("active");
  if (statsPanel) statsPanel.classList.add("active");

  const numCells = rows * cols;
  let recentWins = 0;

  // LOOP INFINITO DE TREINO
  while (!trainingCanceled) {
    // 1. Verifica Pause
    while (trainingPaused && !trainingCanceled) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (trainingCanceled) break;

    // Atualiza estado do turbo a cada ciclo principal
    silentMode = turboCheckbox ? turboCheckbox.checked : false;

    const gamesInBatch = silentMode ? 2000 : 1;

    for (let g = 0; g < gamesInBatch; g++) {
      if (trainingCanceled) break; // Sai rápido se cancelou
      if (trainingPaused) break; // Sai agora se pausou (para entrar no loop de espera externo)

      currentCycle++;
      startGame(activeDiff); // Reseta tabuleiro

      // Variáveis locais da partida
      let episodeReward = 0;
      let moves = 0;
      let won = false;

      // Loop da Partida (Game Loop)
      while (!isGameOver && moves < numCells * 1.5) {
        // 1. Obtém Estado (Cria tensor na memória)
        const state = aiBrain.getState(board);

        // 2. Escolhe Ação (Async)
        const actionIdx = await aiBrain.chooseAction(state);

        const isFlag = actionIdx >= numCells;
        const cellIdx = isFlag ? actionIdx - numCells : actionIdx;
        const r = Math.floor(cellIdx / cols);
        const c = cellIdx % cols;

        // Visualização (Só se NÃO for Turbo)
        if (!silentMode) {
          const targetEl = document.getElementById(`cell-${r}-${c}`);
          const currentTargets = document.getElementsByClassName("ia-target");
          while (currentTargets.length > 0)
            currentTargets[0].classList.remove("ia-target");
          if (targetEl) {
            targetEl.classList.add("ia-target");
            await new Promise((res) => setTimeout(res, 15)); // Delay visual para humano ver
          }
        }

        // 3. Executa Ação e Calcula Recompensa Inteligente
        let reward = 0;
        const cellsRevealedBefore = board
          .flat()
          .filter((c) => c.revealed).length;
        const currentFlags = board.flat().filter((c) => c.flagged).length;

        if (isFlag) {
          // Calculamos o multiplicador de progressão baseado no número de bandeiras
          // Isso garante que o peso da decisão aumente conforme o jogo avança
          const flagMultiplier = 5 * currentFlags;

          if (board[r][c].flagged) {
            const wasCorrect = board[r][c].mine;
            handleRightClick(r, c); // Desmarcar

            if (wasCorrect) {
              // PUNIÇÃO PROGRESSIVA: Retirar uma bandeira que estava CERTA
              // Quanto mais bandeiras no jogo, mais grave é o erro de tirar uma correta
              reward = -(60 + flagMultiplier);
            } else {
              // RECOMPENSA PROGRESSIVA: Tirar uma bandeira que estava ERRADA
              // Premiamos a IA por "perceber" o erro e corrigi-lo
              reward = 15 + Math.floor(flagMultiplier / 2);
            }
          }
        } else {
          const result = handleClick(r, c);
          const cellsRevealedAfter = board
            .flat()
            .filter((c) => c.revealed).length;
          const cellsRevealed = cellsRevealedAfter - cellsRevealedBefore;

          if (result === "mine") {
            reward = -100; // Morte é MUITO grave (antes: -500)
          } else if (result === "safe") {
            // REBALANCEADO: Recompensa proporcional ao progresso
            reward = 10 + cellsRevealed * 3;
          } else if (result === "win") {
            // REBALANCEADO: Bônus de vitória + bônus por eficiência
            const efficiency = numCells / Math.max(moves, 1);
            reward = 2000 + efficiency * 100;
            won = true;
          } else {
            reward = -10; // Movimento inválido (antes: -20)
          }
        }

        // 4. Obtém Próximo Estado
        const nextState = aiBrain.getState(board);

        // 5. Memoriza
        // O tensor 'state' e 'nextState' são entregues ao 'remember'.
        // Eles SÓ serão descartados quando o buffer de memória encher (ver método remember).
        aiBrain.remember(state, actionIdx, reward, nextState, isGameOver);

        episodeReward += reward;
        moves++;

        // 6. Treina (Replay)
        // No modo turbo: Treina a cada 10 passos para speed. No normal: a cada passo.
        if (!silentMode || moves % 10 === 0) {
          await aiBrain.replay();
        }

        // Se visual, libera frame
        if (!silentMode) await tf.nextFrame();
      }

      // Fim da partida
      aiBrain.decayEpsilon();

      if (won) {
        totalWins++;
        recentWins++;
        console.log(
          `%c 🏆 VITÓRIA NO CICLO ${currentCycle}! (Eps: ${aiBrain.epsilon.toFixed(
            3
          )})`,
          "color: #2ecc71; font-weight: bold; font-size: 14px; background: #000; padding: 4px;"
        );
        if (silentMode) {
          if (totalWins % 100 === 0) saveBrainToStorage();
        } else {
          saveBrainToStorage();
        }
      }

      // Log Periódico no console para não floodar
      if (currentCycle % 100 === 0) {
        console.log(
          `Ciclo ${currentCycle} | Wins: ${totalWins} | Epsilon: ${aiBrain.epsilon.toFixed(
            3
          )}`
        );
      }
    } // Fim do For (Batch)

    // === RESUMO DO LOTE (Aparece a cada atualização de tela - 500 jogos) ===
    console.log(
      `[LOTE] Ciclo ${currentCycle} | Total Wins: ${totalWins} | Recentes: ${recentWins} | Epsilon: ${aiBrain.epsilon.toFixed(
        4
      )}`
    );
    recentWins = 0; // Reseta contador de wins recentes

    // Atualiza UI apenas uma vez por Batch
    updateStats();
    if (statusEl)
      statusEl.innerText = `Ciclo ${currentCycle} | Vitórias: ${totalWins} (Turbo: ${
        silentMode ? "ON" : "OFF"
      })`;

    // Libera a thread para a UI não travar totalmente
    await tf.nextFrame();

    // Salva periodicamente (A cada ~2500 partidas)
    if (currentCycle % 2500 < gamesInBatch) {
      console.log(`✅ Backup Automático (Ciclo ${currentCycle})`);
      await saveBrainToStorage();
    }
  }

  // Finalização
  await saveBrainToStorage();
  if (statusEl) statusEl.innerText = `IA Pausada - ${totalWins} vitórias`;
  if (pauseBtn) pauseBtn.style.display = "none";
  trainingMode = false;
  silentMode = false;
  alert(`Treino finalizado!\nCiclos: ${currentCycle}\nVitórias: ${totalWins}`);
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
