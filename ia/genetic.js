const POPULATION_SIZE = 200; // População por geração
const THREADS = Math.min(navigator.hardwareConcurrency || 4, 12); // Workers paralelos
const ELITE_SIZE = 10; // Top 5 melhores IAs mantidas entre gerações

class GeneticManager {
  constructor() {
    this.workers = [];
    this.generation = 0;
    this.isTraining = false;
    this.elitePopulation = []; // Top 5 melhores IAs de todas as gerações
    this.bestScore = -Infinity;
    this.generationHistory = []; // Histórico de performance
  }

  // Crossover: Combina dois pais para criar um filho
  crossover(parent1, parent2) {
    return parent1.map((w, layerIdx) => {
      const p1Data = parent1[layerIdx].data;
      const p2Data = parent2[layerIdx].data;
      const childData = new Float32Array(p1Data.length);

      // Crossover uniforme: cada peso vem de um dos pais
      for (let i = 0; i < p1Data.length; i++) {
        childData[i] = Math.random() < 0.5 ? p1Data[i] : p2Data[i];
      }

      return { data: childData, shape: w.shape };
    });
  }

  // Mutação com taxa adaptativa
  applyMutation(weights, rate = 0.15, amount = 0.2) {
    return weights.map((w) => {
      const newData = new Float32Array(w.data);
      for (let i = 0; i < newData.length; i++) {
        if (Math.random() < rate) {
          // Mutação gaussiana para mudanças mais suaves
          newData[i] += (Math.random() - 0.5) * 2 * amount;
        }
      }
      return { data: newData, shape: w.shape };
    });
  }

  async init() {
    if (!aiBrain || !aiBrain.model) {
      alert("Inicie um jogo primeiro!");
      return;
    }

    console.log(
      `🚀 Iniciando ${THREADS} Workers com Elite de ${ELITE_SIZE}...`,
    );

    // Prepara topologia e pesos iniciais
    const topology = aiBrain.model.toJSON(null, false);
    const weights = [];
    const wTensors = aiBrain.model.getWeights();
    for (let t of wTensors)
      weights.push({ data: await t.data(), shape: t.shape });

    // Apenas prepara os pesos, mas não insere na elite ainda
    // A elite será formada pelos resultados da primeira geração
    this.initialWeights = weights;
    this.elitePopulation = []; // Começa vazia para ser preenchida pelos melhores da Gen 1

    // Cria Pool de Workers
    const readyPromises = [];
    for (let i = 0; i < THREADS; i++) {
      const w = new Worker("worker.js");
      this.workers.push({ worker: w, id: i });

      const readyP = new Promise((resolve) => {
        const handler = (ev) => {
          if (ev.data.type === "READY") {
            w.removeEventListener("message", handler);
            resolve();
          } else if (ev.data.type === "ERROR") {
            console.error(`❌ Worker ${i} falhou:`, ev.data.error);
          }
        };
        w.addEventListener("message", handler);
      });
      readyPromises.push(readyP);

      w.postMessage({
        type: "INIT_MODEL",
        payload: { topology, weights },
      });
    }

    await Promise.all(readyPromises);
    console.log("✅ Todos os Workers prontos!");
  }

  async evolve() {
    if (this.workers.length === 0) await this.init();

    const gamesPerWorker = Math.ceil(POPULATION_SIZE / THREADS);
    const promises = [];

    console.log(
      `🏁 Gen ${this.generation} | ${THREADS} threads × ${gamesPerWorker} jogos = ${POPULATION_SIZE} indivíduos`,
    );

    // Distribui trabalho entre workers
    // Cada worker recebe uma mistura de: elite, crossovers e mutações
    for (let i = 0; i < THREADS; i++) {
      const p = new Promise((resolve, reject) => {
        const w = this.workers[i].worker;

        const handler = (ev) => {
          if (ev.data.type === "BATCH_DONE") {
            w.removeEventListener("message", handler);
            resolve(ev.data.results);
          } else if (ev.data.type === "ERROR") {
            console.error(`💥 Erro no Worker ${i}:`, ev.data.error);
            w.removeEventListener("message", handler);
            reject(ev.data.error);
          }
        };

        w.addEventListener("message", handler);

        // Envia população diversificada para o worker
        const eliteWeights =
          this.elitePopulation.length > 0
            ? this.elitePopulation[0].weights
            : this.elitePopulation[0]?.weights || null;

        w.postMessage({
          type: "RUN_BATCH",
          payload: {
            gamesToPlay: gamesPerWorker,
            elitePopulation:
              this.elitePopulation.length > 0
                ? this.elitePopulation.map((e) => e.weights)
                : [this.initialWeights], // Envia mestre como semente
            eliteSize: ELITE_SIZE, // ✅ Envia o tamanho da elite
            rows: aiBrain.rows,
            cols: aiBrain.cols,
            mines: mines,
          },
        });
      });
      promises.push(p);
    }

    // Aguarda todos os workers
    const resultsArrays = await Promise.all(promises);
    const allResults = resultsArrays.flat();

    // Ordena por score
    allResults.sort((a, b) => b.score - a.score);

    // Atualiza elite (Top 5 de TODAS as gerações)
    const newCandidates = allResults.slice(0, ELITE_SIZE).map((r) => ({
      weights: r.weights,
      score: r.score,
      victory: r.victory,
      generation: this.generation,
    }));

    // Combina elite antiga com novos candidatos e pega os top 5
    const combinedElite = [...this.elitePopulation, ...newCandidates]
      .filter((e) => e.weights !== null) // Remove entradas sem pesos
      .sort((a, b) => b.score - a.score)
      .slice(0, ELITE_SIZE);

    this.elitePopulation = combinedElite;

    // Melhor de todos os tempos
    const best = this.elitePopulation[0];
    const generationBest = allResults[0];

    // Atualiza modelo principal com o melhor de todos os tempos
    if (best.weights) {
      const newWeights = best.weights.map((w) => tf.tensor(w.data, w.shape));
      aiBrain.model.setWeights(newWeights);
      newWeights.forEach((t) => t.dispose());
    }

    // Registra histórico
    this.generationHistory.push({
      generation: this.generation,
      bestScore: generationBest.score,
      avgScore:
        allResults.reduce((sum, r) => sum + r.score, 0) / allResults.length,
      victories: allResults.filter((r) => r.victory).length,
      eliteScore: best.score,
    });

    // Atualiza UI
    const statusEl = document.getElementById("ia-status");
    if (statusEl) {
      const top5Scores = this.elitePopulation
        .map((e) => e.score.toFixed(0))
        .join(", ");
      statusEl.innerText = `Gen ${this.generation} | Top5: [${top5Scores}]`;
    }

    const cycleEl = document.getElementById("cycle-count");
    if (cycleEl) cycleEl.innerText = `Gen ${this.generation}`;

    // Console com Top 5
    console.log(`\n${"=".repeat(60)}`);
    console.log(`📊 GERAÇÃO ${this.generation} - Resumo:`);
    console.log(`${"=".repeat(60)}`);
    console.log(`🏆 TOP 5 ELITE (de todas as gerações):`);
    this.elitePopulation.forEach((elite, idx) => {
      const medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][idx];
      const victoryIcon = elite.victory ? "✅" : "❌";
      console.log(
        `  ${medal} Score: ${elite.score.toFixed(1).padStart(8)} ${victoryIcon} (Gen ${elite.generation})`,
      );
    });

    console.log(`\n📈 Esta Geração:`);
    console.log(
      `  Melhor: ${generationBest.score.toFixed(1)} ${generationBest.victory ? "🏆" : "💀"}`,
    );
    console.log(
      `  Média: ${(allResults.reduce((s, r) => s + r.score, 0) / allResults.length).toFixed(1)}`,
    );
    console.log(
      `  Vitórias: ${allResults.filter((r) => r.victory).length}/${POPULATION_SIZE}`,
    );
    console.log(`${"=".repeat(60)}\n`);

    // Salva se houver vitória
    if (generationBest.victory) {
      totalWins++;
      updateStats();
      saveBrainToStorage();
    }

    this.generation++;
  }

  stop() {
    this.workers.forEach((w) => w.worker.terminate());
    this.workers = [];

    // Mostra resumo final
    if (this.generationHistory.length > 0) {
      console.log("\n📊 RESUMO DO TREINAMENTO:");
      console.log(`Gerações: ${this.generationHistory.length}`);
      console.log(`Melhor Score: ${this.elitePopulation[0].score.toFixed(1)}`);
      console.log(`Vitórias Totais: ${totalWins}`);
    }
  }
}

// Global hook
let manager = null;

async function startGeneticTraining() {
  const btn = document.getElementById("btn-genetic");

  if (manager && manager.isTraining) {
    manager.isTraining = false;
    manager.stop();
    btn.innerText = "🧬 Escolinha Genética";
    btn.style.background = "#9b59b6";
    return;
  }

  btn.innerText = "🛑 Parar (Genético)";
  btn.style.background = "#c0392b";

  manager = new GeneticManager();
  manager.isTraining = true;

  while (manager.isTraining) {
    await manager.evolve();
    await new Promise((r) => setTimeout(r, 100)); // Pausa para UI
  }
}
