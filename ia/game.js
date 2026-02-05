// game.js - Lógica Principal do Jogo
let board = [];
let rows, cols, mines;
let timerInterval;
let seconds = 0;
let isGameOver = false;
let activeDiff = "";
let trainingMode = false;
let silentMode = false;

const config = {
  facil: { r: 9, c: 9, m: 10 },
  medio: { r: 16, c: 16, m: 40 },
  dificil: { r: 16, c: 30, m: 99 },
};

function startGame(level) {
  activeDiff = level;

  // Se for "custom", usa as variáveis globais já configuradas externamente (pelo Curriculum)
  if (level !== "custom") {
    const c = config[level];
    rows = c.r;
    cols = c.c;
    mines = c.m;
  }

  const diffLabel = document.getElementById("diff-label");
  if (diffLabel) diffLabel.innerText = level.toUpperCase();

  resetGame();
  renderRanking(); // Renderiza o ranking ao trocar de dificuldade
  if (!trainingMode) closeModal();
}

function resetGame() {
  clearInterval(timerInterval);
  seconds = 0;
  isGameOver = false;

  // OTIMIZAÇÃO: No modo silencioso, pula TODA a manipulação DOM
  if (!silentMode) {
    const timerEl = document.getElementById("timer");
    const mineEl = document.getElementById("mine-count");
    if (timerEl) timerEl.innerText = "0";
    if (mineEl) mineEl.innerText = mines;

    const grid = document.getElementById("grid");
    if (grid) {
      grid.innerHTML = "";
      // Tamanho dinâmico baseado no número de colunas
      const cellSize = cols <= 9 ? 40 : cols <= 16 ? 32 : 24;
      grid.style.gridTemplateColumns = `repeat(${cols}, ${cellSize}px)`;
      grid.style.gridTemplateRows = `repeat(${rows}, ${cellSize}px)`;
    }
  }

  board = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => ({
      mine: false,
      revealed: false,
      flagged: false,
      count: 0,
    })),
  );

  // Plantar minas
  let mPlanted = 0;
  while (mPlanted < mines) {
    let r = Math.floor(Math.random() * rows);
    let c = Math.floor(Math.random() * cols);
    if (!board[r][c].mine) {
      board[r][c].mine = true;
      mPlanted++;
    }
  }

  // Calcular números vizinhos
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (board[r][c].mine) continue;
      let count = 0;
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
          if (board[r + i]?.[c + j]?.mine) count++;
        }
      }
      board[r][c].count = count;
    }
  }

  // OTIMIZAÇÃO: Renderizar células apenas se NÃO estiver em modo silencioso
  if (!silentMode) {
    const grid = document.getElementById("grid");
    if (grid) {
      const cellSize = cols <= 9 ? 40 : cols <= 16 ? 32 : 24;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const el = document.createElement("div");
          el.classList.add("cell");
          el.id = `cell-${r}-${c}`;
          el.style.minWidth = `${cellSize}px`;
          el.style.minHeight = `${cellSize}px`;
          el.addEventListener("click", () => handleClick(r, c));
          el.addEventListener("contextmenu", (e) => {
            e.preventDefault();
            handleRightClick(r, c);
          });
          grid.appendChild(el);
        }
      }
    }
  }
}

function handleRightClick(r, c) {
  if (isGameOver || board[r][c].revealed) return;

  // REMOVIDO: Limite artificial de bandeiras que bloqueava a IA
  // A IA agora pode marcar/desmarcar livremente para corrigir erros

  const el = document.getElementById(`cell-${r}-${c}`);
  board[r][c].flagged = !board[r][c].flagged;

  if (silentMode) return; // Evita processamento de interface

  if (el) {
    el.classList.toggle("flagged");
    el.innerText = board[r][c].flagged ? "🚩" : "";

    // Atualiza contador de minas visual
    const mineEl = document.getElementById("mine-count");
    if (mineEl)
      mineEl.innerText =
        mines - board.flat().filter((cell) => cell.flagged).length;
  }
}

function handleClick(r, c) {
  if (isGameOver || board[r][c].revealed || board[r][c].flagged)
    return "invalid";

  // Remove marcação de "pensamento" quando o humano ou a IA clica (apenas modo visual)
  if (!silentMode) {
    const el = document.getElementById(`cell-${r}-${c}`);
    if (el) el.classList.remove("ia-target");
  }

  if (seconds === 0 && !trainingMode) startTimer();

  if (board[r][c].mine) {
    revealAllMines();
    if (!trainingMode) showModal("lose");
    return "mine";
  }

  reveal(r, c);

  if (checkWin()) {
    if (!trainingMode) {
      saveRecord(seconds); // Salva o recorde se ganhar
      showModal("win");
    }
    return "win";
  }
  return "safe";
}

function reveal(r, c) {
  if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c].revealed) return;

  board[r][c].revealed = true;

  if (silentMode) {
    // No modo silencioso, apenas processa a lógica de cascata sem mexer no HTML
    if (board[r][c].count === 0) {
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) reveal(r + i, c + j);
      }
    }
    return;
  }

  const el = document.getElementById(`cell-${r}-${c}`);

  if (el) {
    el.classList.add("revealed");
    if (board[r][c].count > 0) {
      el.innerText = board[r][c].count;
      el.classList.add(`n${board[r][c].count}`);
    } else {
      for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) reveal(r + i, c + j);
      }
    }
  }
}

function checkWin() {
  const revealedCount = board.flat().filter((c) => c.revealed).length;
  if (revealedCount === rows * cols - mines) {
    isGameOver = true;
    clearInterval(timerInterval);
    return true;
  }
  return false;
}

function revealAllMines() {
  isGameOver = true;
  clearInterval(timerInterval);

  // No modo silencioso, não precisa revelar visualmente
  if (silentMode) return;

  board.forEach((row, r) =>
    row.forEach((cell, c) => {
      if (cell.mine) {
        const el = document.getElementById(`cell-${r}-${c}`);
        if (el) {
          el.classList.add("mine", "revealed");
          el.innerText = "💣";
        }
      }
    }),
  );
}

function startTimer() {
  timerInterval = setInterval(() => {
    seconds++;
    const timerEl = document.getElementById("timer");
    if (timerEl) timerEl.innerText = seconds;
  }, 1000);
}

function showModal(type) {
  const overlay = document.getElementById("modal-overlay");
  const title = document.getElementById("modal-title");
  const msg = document.getElementById("modal-msg");

  if (overlay) overlay.style.display = "flex";
  if (title) title.innerText = type === "win" ? "🔥 VITÓRIA!" : "💥 BUM!";
  if (msg)
    msg.innerText = type === "win" ? `Tempo: ${seconds}s` : "Tente novamente!";
}

function closeModal() {
  const overlay = document.getElementById("modal-overlay");
  if (overlay) overlay.style.display = "none";
  if (isGameOver && !trainingMode) resetGame();
}

// --- SISTEMA DE RANKING ---
function saveRecord(time) {
  const records =
    JSON.parse(localStorage.getItem(`minesweeper_ranking_${activeDiff}`)) || [];
  const newRecord = {
    time: time,
    date: new Date().toLocaleDateString("pt-BR"),
  };

  records.push(newRecord);
  records.sort((a, b) => a.time - b.time);

  // Mantém apenas o Top 10
  const top10 = records.slice(0, 10);
  localStorage.setItem(
    `minesweeper_ranking_${activeDiff}`,
    JSON.stringify(top10),
  );
  renderRanking();
}

function renderRanking() {
  const tbody = document.querySelector("#leaderboard tbody");
  if (!tbody) return;

  const records =
    JSON.parse(localStorage.getItem(`minesweeper_ranking_${activeDiff}`)) || [];

  if (records.length === 0) {
    tbody.innerHTML =
      '<tr><td colspan="3">Sem recordes ainda nesta dificuldade</td></tr>';
    return;
  }

  tbody.innerHTML = records
    .map(
      (rec, index) => `
        <tr>
            <td><strong>#${index + 1}</strong></td>
            <td>${rec.time}s</td>
            <td>${rec.date}</td>
        </tr>
    `,
    )
    .join("");
}
