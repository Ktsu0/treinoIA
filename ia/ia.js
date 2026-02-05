// ia.js - Versão Corrigida e Otimizada com Sistema de Persistência
var aiBrain;
var trainingPaused = false;
var trainingCanceled = false;
var currentCycle = 0;
var totalWins = 0;

class MinesweeperAI {
  constructor(rows = 9, cols = 9) {
    // Garante que são números válidos
    rows = rows || 9;
    cols = cols || 9;

    // Define o tamanho MÁXIMO do cérebro (o objetivo final)
    // Se receber rows/cols menores, assumimos que é o tamanho atual do jogo,
    // mas fixamos a arquitetura interna em pelo menos 9x9 para garantir consistência.
    this.maxRows = Math.max(rows, 9);
    this.maxCols = Math.max(cols, 9);

    // Tamanho ATUAL do jogo (pode mudar dinamicamente)
    this.currentRows = rows;
    this.currentCols = cols;

    // Mantém compatibilidade com código que acessa .rows/.cols
    this.rows = rows;
    this.cols = cols;

    this.gamma = 0.95;
    this.learningRate = 0.003;
    this.epsilon = 1.0;
    this.epsilonMin = 0.2;
    this.epsilonDecay = 0.9995;
    this.memory = [];
    this.maxMemory = 10000;
    this.batchSize = 32;
    this.model = this.createModel();
  }

  createModel() {
    const model = tf.sequential();
    model.add(
      tf.layers.conv2d({
        inputShape: [this.maxRows, this.maxCols, 3], // Entrada FIXA no tamanho máximo
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
        units: this.maxRows * this.maxCols * 2, // Saída FIXA no tamanho máximo
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
      // Cria buffer do tamanho MÁXIMO
      const numCellsMax = this.maxRows * this.maxCols;
      const buffer = new Float32Array(numCellsMax * 3).fill(-1); // -1 = Fora do tabuleiro

      const flatBoard = board.flat();

      // Preenche apenas a área ativa do jogo atual
      for (let r = 0; r < this.currentRows; r++) {
        for (let c = 0; c < this.currentCols; c++) {
          // Índice no tabuleiro pequeno
          const iSmall = r * this.currentCols + c;
          // Índice mapeado para o buffer GRANDE (alinhado ao topo-esquerda)
          const iBig = r * this.maxCols + c;

          const cell = flatBoard[iSmall];
          const base = iBig * 3;

          let value = 0;
          if (cell.revealed) value = (cell.count + 1) / 9;

          const hidden = cell.revealed ? 0 : 1;
          const flagged = cell.flagged ? 1 : 0;

          buffer[base] = value;
          buffer[base + 1] = hidden;
          buffer[base + 2] = flagged;
        }
      }

      return tf.tensor3d(buffer, [this.maxRows, this.maxCols, 3]).expandDims(0);
    });
  }

  async chooseAction(state) {
    const flatBoard = board.flat();

    // EXPLORAÇÃO
    if (Math.random() < this.epsilon) {
      // Escolhe apenas entre células válidas do tabuleiro ATUAL
      const validActions = [];
      const numCellsCurrent = this.currentRows * this.currentCols;

      for (let i = 0; i < numCellsCurrent; i++) {
        if (!flatBoard[i].revealed) {
          validActions.push(i); // Clique
          validActions.push(i + numCellsCurrent); // Bandeira (offset relativo ao atual)
        }
      }

      if (validActions.length === 0) return 0; // Fallback

      const actionSmall =
        validActions[Math.floor(Math.random() * validActions.length)];

      // Retorna o índice "pequeno" para o jogo usar diretamente
      // (O jogo não sabe sobre o cérebro grande)
      return actionSmall;
    }

    // EXPLORAÇÃO COM REDE
    return tf.tidy(() => {
      const prediction = this.model.predict(state);
      const predData = prediction.dataSync();

      let bestActionSmall = 0; // Este é o índice que o jogo entende
      let bestValue = -Infinity;

      const numCellsMax = this.maxRows * this.maxCols;
      const numCellsCurrent = this.currentRows * this.currentCols;

      // Varre apenas a área válida do tabuleiro atual
      for (let r = 0; r < this.currentRows; r++) {
        for (let c = 0; c < this.currentCols; c++) {
          const iSmall = r * this.currentCols + c; // Índice no jogo
          const iBig = r * this.maxCols + c; // Índice na rede neural

          if (!flatBoard[iSmall].revealed) {
            // Verificar output de CLIQUE da rede
            const valClick = predData[iBig];
            if (valClick > bestValue) {
              bestValue = valClick;
              bestActionSmall = iSmall;
            }

            // Verificar output de BANDEIRA da rede
            // O offset de bandeira na rede é numCellsMax
            const valFlag = predData[iBig + numCellsMax];
            if (valFlag > bestValue) {
              bestValue = valFlag;
              // O offset de bandeira para o jogo é numCellsCurrent
              bestActionSmall = iSmall + numCellsCurrent;
            }
          }
        }
      }

      return bestActionSmall;
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
        // Salva estado do Curriculum se disponível
        curriculumLevel:
          typeof curriculum !== "undefined" ? curriculum.currentLevel : 0,
        curriculumWins:
          typeof curriculum !== "undefined" ? curriculum.levelWins : 0,
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

    const gamesInBatch = silentMode ? 5000 : 1; // Aumentado de 2000 para 5000

    for (let g = 0; g < gamesInBatch; g++) {
      if (trainingCanceled) break; // Sai rápido se cancelou
      if (trainingPaused) break; // Sai agora se pausou (para entrar no loop de espera externo)

      currentCycle++;

      // CORREÇÃO: Sempre chama startGame no primeiro jogo para inicializar variáveis
      // Depois, no modo turbo, usa resetGame para performance
      if (g === 0 || !silentMode) {
        startGame(activeDiff); // Inicializa/reseta com visualização
      } else {
        resetGame(); // Apenas reseta lógica (modo turbo)
      }

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

        // 3. Executa Ação e Calcula Recompensa
        let reward = 0;
        const cellsRevealedBefore = board
          .flat()
          .filter((c) => c.revealed).length;

        if (isFlag) {
          // Ação de BANDEIRA
          const wasFlagged = board[r][c].flagged;
          handleRightClick(r, c); // Toggle bandeira
          const isMine = board[r][c].mine;

          if (!wasFlagged) {
            // MARCOU uma bandeira
            if (isMine) {
              reward = 50; // ✅ Acertou! Marcou uma mina
            } else {
              reward = -30; // ❌ Errou! Marcou uma célula segura
            }
          } else {
            // DESMARCOU uma bandeira
            if (isMine) {
              reward = -60; // ❌ Erro grave! Desmarcou uma mina correta
            } else {
              reward = 20; // ✅ Bom! Corrigiu um erro
            }
          }
        } else {
          // Ação de REVELAR
          const result = handleClick(r, c);
          const cellsRevealedAfter = board
            .flat()
            .filter((c) => c.revealed).length;
          const cellsRevealed = cellsRevealedAfter - cellsRevealedBefore;

          if (result === "mine") {
            reward = -100; // ❌ Morte
          } else if (result === "safe") {
            reward = 10 + cellsRevealed * 5; // ✅ Revelou células (mais = melhor)
          } else if (result === "win") {
            const efficiency = numCells / Math.max(moves, 1);
            reward = 2000 + efficiency * 100; // 🏆 VITÓRIA!
            won = true;
          } else {
            reward = -5; // Movimento inválido (já revelada ou marcada)
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
        // No modo turbo: Treina a cada 50 passos para speed máximo. No normal: a cada passo.
        if (!silentMode) {
          await aiBrain.replay();
          await tf.nextFrame(); // Libera frame apenas no modo visual
        } else if (moves % 50 === 0) {
          // No modo turbo, treina muito menos frequentemente
          await aiBrain.replay();
        }
      }

      // Fim da partida
      aiBrain.decayEpsilon();

      if (won) {
        totalWins++;
        recentWins++;
        console.log(
          `%c 🏆 VITÓRIA NO CICLO ${currentCycle}! (Eps: ${aiBrain.epsilon.toFixed(
            3,
          )})`,
          "color: #2ecc71; font-weight: bold; font-size: 14px; background: #000; padding: 4px;",
        );
        // OTIMIZAÇÃO: Salva muito menos frequentemente no modo turbo
        if (silentMode) {
          if (totalWins % 500 === 0) saveBrainToStorage(); // A cada 500 vitórias
        } else {
          saveBrainToStorage();
        }
      }

      // Log Periódico no console para não floodar
      if (currentCycle % 100 === 0) {
        const avgReward = episodeReward / Math.max(moves, 1);
        console.log(
          `Ciclo ${currentCycle} | Wins: ${totalWins} | Epsilon: ${aiBrain.epsilon.toFixed(
            3,
          )} | Reward: ${episodeReward.toFixed(1)} | Avg: ${avgReward.toFixed(2)}`,
        );
      }
    } // Fim do For (Batch)

    // === RESUMO DO LOTE (Aparece a cada atualização de tela - 500 jogos) ===
    console.log(
      `[LOTE] Ciclo ${currentCycle} | Total Wins: ${totalWins} | Recentes: ${recentWins} | Epsilon: ${aiBrain.epsilon.toFixed(
        4,
      )}`,
    );
    recentWins = 0; // Reseta contador de wins recentes

    // Atualiza UI apenas uma vez por Batch
    updateStats();
    if (statusEl)
      statusEl.innerText = `Ciclo ${currentCycle} | Vitórias: ${totalWins} (Turbo: ${
        silentMode ? "ON" : "OFF"
      })`;

    // OTIMIZAÇÃO: Libera a thread apenas no modo visual
    if (!silentMode) {
      await tf.nextFrame();
    } else {
      // No modo turbo, apenas um micro-delay para não travar completamente o navegador
      await new Promise((r) => setTimeout(r, 0));
    }

    // Salva periodicamente (A cada ~5000 partidas no turbo, 2500 no normal)
    const saveInterval = silentMode ? 5000 : 2500;
    if (currentCycle % saveInterval < gamesInBatch) {
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
