// curriculum.js - Sistema de Aprendizado Progressivo

class CurriculumTraining {
  constructor() {
    this.currentLevel = 0;
    this.levels = [
      { name: "Baby", rows: 3, cols: 3, mines: 1, winsNeeded: 10 },
      { name: "Criança", rows: 4, cols: 4, mines: 2, winsNeeded: 20 },
      { name: "Jovem", rows: 5, cols: 5, mines: 4, winsNeeded: 30 },
      { name: "Adulto", rows: 6, cols: 6, mines: 6, winsNeeded: 40 },
      { name: "Expert", rows: 7, cols: 7, mines: 8, winsNeeded: 50 },
      { name: "Master", rows: 9, cols: 9, mines: 10, winsNeeded: 100 },
    ];
    this.levelWins = 0;
  }

  getCurrentLevel() {
    return this.levels[this.currentLevel];
  }

  recordWin() {
    this.levelWins++;
    const level = this.getCurrentLevel();

    if (
      this.levelWins >= level.winsNeeded &&
      this.currentLevel < this.levels.length - 1
    ) {
      this.currentLevel++;
      this.levelWins = 0;
      console.log(`\n${"🎓".repeat(50)}`);
      console.log(
        `🎓 NÍVEL COMPLETO! Avançando para: ${this.levels[this.currentLevel].name}`,
      );
      console.log(
        `🎓 Novo desafio: ${this.levels[this.currentLevel].rows}x${this.levels[this.currentLevel].cols} com ${this.levels[this.currentLevel].mines} minas`,
      );
      console.log(`${"🎓".repeat(50)}\n`);
      return true; // Mudou de nível
    }
    return false; // Mesmo nível
  }

  getProgress() {
    const level = this.getCurrentLevel();
    const percentage = ((this.levelWins / level.winsNeeded) * 100).toFixed(1);
    return `${level.name} (${this.levelWins}/${level.winsNeeded} - ${percentage}%)`;
  }

  reset() {
    this.currentLevel = 0;
    this.levelWins = 0;
  }
}

// Instância global
let curriculum = new CurriculumTraining();

// Função para iniciar treino com curriculum
async function trainWithCurriculum() {
  const statusEl = document.getElementById("ia-status");
  const pauseBtn = document.getElementById("pause-btn");

  if (statusEl) statusEl.innerText = "Modo: Curriculum Learning...";
  if (pauseBtn) pauseBtn.style.display = "inline-block";

  trainingMode = true;
  trainingPaused = false;
  trainingCanceled = false;

  const turboCheckbox = document.getElementById("turbo-mode");
  silentMode = turboCheckbox ? turboCheckbox.checked : false;

  // Tenta carregar estado salvo
  const savedMeta = JSON.parse(
    localStorage.getItem("minesweeper-metadata") || "{}",
  );
  if (savedMeta.curriculumLevel !== undefined) {
    curriculum.currentLevel = savedMeta.curriculumLevel;
    curriculum.levelWins = savedMeta.curriculumWins || 0;
    console.log(
      `🔄 Curriculum restaurado: Nível ${curriculum.getCurrentLevel().name}`,
    );
  } else {
    // Se não tiver save, começa do zero
    curriculum.reset();
  }

  const progressContainer = document.querySelector(".training-progress");
  const statsPanel = document.querySelector(".training-stats");

  if (progressContainer) progressContainer.classList.add("active");
  if (statsPanel) statsPanel.classList.add("active");

  console.log("\n🎓 INICIANDO CURRICULUM LEARNING 🎓");
  console.log("Estratégia: Começar fácil e aumentar gradualmente\n");

  // LOOP INFINITO DE TREINO COM CURRICULUM
  while (!trainingCanceled) {
    // Verifica Pause
    while (trainingPaused && !trainingCanceled) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (trainingCanceled) break;

    // Atualiza estado do turbo
    silentMode = turboCheckbox ? turboCheckbox.checked : false;

    // Pega nível atual
    const level = curriculum.getCurrentLevel();

    // Atualiza configuração do jogo
    rows = level.rows;
    cols = level.cols;
    mines = level.mines;

    // Recria IA se necessário ou ATUALIZA parâmetros se for curriculum
    if (!aiBrain) {
      // Cria a IA pela primeira vez com o tamanho MÁXIMO do desafio final
      // Isso é crucial para o Transfer Learning funcionar (pesos fixos)
      console.log(`🧠 Criando Cérebro Mestre (max 9x9)...`);
      aiBrain = new MinesweeperAI(9, 9); // Tamanho máximo fixo

      // Define o tamanho atual para o nível (ex: 3x3)
      aiBrain.currentRows = rows;
      aiBrain.currentCols = cols;

      if (!aiBrain.model.optimizer) {
        aiBrain.model.compile({
          optimizer: tf.train.adam(aiBrain.learningRate),
          loss: "meanSquaredError",
        });
      }
    } else {
      // Apenas ATUALIZA o tamanho virtual do jogo
      // O cérebro físico continua sendo 9x9
      aiBrain.currentRows = rows;
      aiBrain.currentCols = cols;
      // Atualiza também as propriedades de compatibilidade (agora getters no ia.js cuidam disso)
      console.log(`🔄 Adaptando cérebro para ${rows}x${cols} (Input Padding)`);
    }

    const gamesInBatch = silentMode ? 1000 : 1;

    for (let g = 0; g < gamesInBatch; g++) {
      if (trainingCanceled) break;
      if (trainingPaused) break;

      currentCycle++;

      // Reseta jogo
      if (g === 0 || !silentMode) {
        startGame("custom"); // ✅ Usa as dimensões 3x3 definidas acima
      } else {
        resetGame();
      }

      // Variáveis locais da partida
      let episodeReward = 0;
      let moves = 0;
      let won = false;
      const numCells = rows * cols;

      // Loop da Partida
      while (!isGameOver && moves < numCells * 1.5) {
        const state = aiBrain.getState(board);
        const actionIdx = await aiBrain.chooseAction(state);

        const isFlag = actionIdx >= numCells;
        const cellIdx = isFlag ? actionIdx - numCells : actionIdx;
        const r = Math.floor(cellIdx / cols);
        const c = cellIdx % cols;

        // Visualização (só se não for Turbo)
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

        // Executa Ação
        let reward = 0;
        const cellsRevealedBefore = board
          .flat()
          .filter((c) => c.revealed).length;

        if (isFlag) {
          const wasFlagged = board[r][c].flagged;
          handleRightClick(r, c);
          const isMine = board[r][c].mine;

          if (!wasFlagged) {
            reward = isMine ? 50 : -30;
          } else {
            reward = isMine ? -60 : 20;
          }
        } else {
          const result = handleClick(r, c);
          const cellsRevealedAfter = board
            .flat()
            .filter((c) => c.revealed).length;
          const cellsRevealed = cellsRevealedAfter - cellsRevealedBefore;

          if (result === "mine") {
            reward = -100;
          } else if (result === "safe") {
            reward = 10 + cellsRevealed * 5;
          } else if (result === "win") {
            const efficiency = numCells / Math.max(moves, 1);
            reward = 2000 + efficiency * 100;
            won = true;
          } else {
            reward = -5;
          }
        }

        const nextState = aiBrain.getState(board);
        aiBrain.remember(state, actionIdx, reward, nextState, isGameOver);
        episodeReward += reward;
        moves++;

        // Treina
        if (!silentMode) {
          await aiBrain.replay();
          await tf.nextFrame();
        } else if (moves % 50 === 0) {
          await aiBrain.replay();
        }
      }

      // Fim da partida
      aiBrain.decayEpsilon();

      if (won) {
        totalWins++;
        const levelChanged = curriculum.recordWin();

        console.log(
          `%c 🏆 VITÓRIA! Ciclo ${currentCycle} | ${curriculum.getProgress()}`,
          "color: #2ecc71; font-weight: bold; font-size: 14px; background: #000; padding: 4px;",
        );

        if (levelChanged) {
          // Salva ao mudar de nível
          saveBrainToStorage();
          // 🚨 IMPORTANTE: Quebra o batch atual para atualizar o tabuleiro!
          // Se não fizer isso, ela continua jogando 3x3 até acabar o lote de 1000 jogos
          break;
        }
      }

      // Log Periódico
      if (currentCycle % 100 === 0) {
        const avgReward = episodeReward / Math.max(moves, 1);
        console.log(
          `Ciclo ${currentCycle} | ${curriculum.getProgress()} | Epsilon: ${aiBrain.epsilon.toFixed(3)} | Avg: ${avgReward.toFixed(2)}`,
        );
      }
    }

    // Atualiza UI
    updateStats();
    if (statusEl)
      statusEl.innerText = `${curriculum.getProgress()} | Ciclo ${currentCycle} | Wins: ${totalWins}`;

    // Libera thread
    if (!silentMode) {
      await tf.nextFrame();
    } else {
      await new Promise((r) => setTimeout(r, 0));
    }

    // Salva periodicamente
    const saveInterval = silentMode ? 5000 : 2500;
    if (currentCycle % saveInterval < gamesInBatch) {
      console.log(`✅ Backup Automático (Ciclo ${currentCycle})`);
      await saveBrainToStorage();
    }
  }

  // Finalização
  await saveBrainToStorage();
  if (statusEl)
    statusEl.innerText = `Curriculum Pausado - ${totalWins} vitórias`;
  if (pauseBtn) pauseBtn.style.display = "none";
  trainingMode = false;
  silentMode = false;
  alert(
    `Treino finalizado!\nCiclos: ${currentCycle}\nVitórias: ${totalWins}\nNível: ${curriculum.getProgress()}`,
  );
}
