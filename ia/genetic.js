const POPULATION_SIZE = 600; // ALTA CARGA: 12 threads x 100 jogos cada = CPU FRITANDO
const THREADS = navigator.hardwareConcurrency || 6; // Usa todos os núcleos disponíveis (ou 6 por padrão)

class GeneticManager {
  constructor() {
    this.workers = [];
    this.generation = 0;
    this.isTraining = false;
    this.bestWeightsData = null; // Stored as simple arrays for easy transfer
  }

  async init() {
    if (!aiBrain || !aiBrain.model) {
      alert("Inicie um jogo primeiro!");
      return;
    }

    console.log(`🚀 Iniciando ${THREADS} Workers...`);

    // Prepara topologia e pesos iniciais
    const topology = aiBrain.model.toJSON(null, false);
    const weights = [];
    const wTensors = aiBrain.model.getWeights();
    for (let t of wTensors)
      weights.push({ data: await t.data(), shape: t.shape });

    this.bestWeightsData = weights; // Pai inicial

    // Cria Pool de Workers
    for (let i = 0; i < THREADS; i++) {
      const w = new Worker("worker.js");
      // Handler temporário até começar o loop
      this.workers.push({ worker: w, id: i });

      w.postMessage({
        type: "INIT_MODEL",
        payload: { topology, weights: this.bestWeightsData },
      });
    }

    // Espera um pouco para garantir init
    await new Promise((r) => setTimeout(r, 1000));
  }

  async evolve() {
    if (this.workers.length === 0) await this.init();

    const gamesPerWorker = Math.ceil(POPULATION_SIZE / THREADS);
    const promises = [];

    console.log(
      `🏁 Gen ${this.generation} | Disparando ${THREADS} threads (${gamesPerWorker} jogos/thread)...`,
    );

    // Dispara tarefas
    for (let i = 0; i < THREADS; i++) {
      const p = new Promise((resolve) => {
        const w = this.workers[i].worker;

        // Handler único para essa rodada
        const handler = (ev) => {
          if (ev.data.type === "BATCH_DONE") {
            w.removeEventListener("message", handler);
            resolve(ev.data.results);
          }
        };
        w.addEventListener("message", handler);

        w.postMessage({
          type: "RUN_BATCH",
          payload: {
            gamesToPlay: gamesPerWorker,
            weights: this.bestWeightsData, // Manda o melhor atual para ser mutado lá dentro
            rows: aiBrain.rows,
            cols: aiBrain.cols,
            mines: mines,
          },
        });
      });
      promises.push(p);
    }

    // Espera todos voltarem (Parallel Join)
    const resultsArrays = await Promise.all(promises);
    const allResults = resultsArrays.flat();

    // Seleção
    allResults.sort((a, b) => b.score - a.score);
    const best = allResults[0];

    // Se o melhor dessa rodada retornou pesos (significa que foi bom), atualizamos
    if (best.weights) {
      this.bestWeightsData = best.weights;
      console.log("🧬 Evolução: Novo melhor cérebro encontrado!");

      // CRÍTICO: Atualiza o cérebro principal com os novos pesos
      const newWeights = this.bestWeightsData.map((w) =>
        tf.tensor(w.data, w.shape),
      );
      aiBrain.model.setWeights(newWeights);
      newWeights.forEach((t) => t.dispose());
    }

    // Atualiza UI
    const statusEl = document.getElementById("ia-status");
    if (statusEl)
      statusEl.innerText = `Gen ${this.generation}: Score ${best.score.toFixed(0)} ${best.victory ? "🏆" : ""}`;

    const cycleEl = document.getElementById("cycle-count");
    if (cycleEl) cycleEl.innerText = `Gen ${this.generation}`;

    // Feedback Visual no Console
    if (best.victory) {
      console.log(
        `%c 🏆 Gen ${this.generation}: SCORE ${best.score.toFixed(1)} (VITÓRIA!) `,
        "background: #2ecc71; color: black; font-weight: bold; padding: 4px; font-size: 14px",
      );

      totalWins++;
      updateStats();
      saveBrainToStorage();
    } else {
      console.log(
        `%c ❌ Gen ${this.generation}: Melhor Score ${best.score.toFixed(1)} (Derrota) `,
        "color: #e74c3c; font-weight: bold;",
      );
    }

    this.generation++;
  }

  stop() {
    this.workers.forEach((w) => w.worker.terminate());
    this.workers = [];
  }
}

// Global hook
let manager = null;

async function startGeneticTraining() {
  const btn = document.getElementById("btn-genetic");

  if (manager && manager.isTraining) {
    manager.isTraining = false;
    manager.stop();
    btn.innerText = "🧬 Escolinha Multi-Core";
    btn.style.background = "#9b59b6";
    return;
  }

  btn.innerText = "🛑 Parar (Multi-Core)";
  btn.style.background = "#c0392b";

  manager = new GeneticManager();
  manager.isTraining = true;

  while (manager.isTraining) {
    await manager.evolve();
    // Pequena pausa para UI respirar e não travar o navegador
    await new Promise((r) => setTimeout(r, 50));
  }
}
