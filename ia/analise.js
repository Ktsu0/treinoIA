const fs = require("fs");
const path = require("path");

function analisarIA(nomeArquivo) {
  const caminhoAbsoluto = path.resolve(__dirname, nomeArquivo);

  if (!fs.existsSync(caminhoAbsoluto)) {
    console.log(`\n❌ Arquivo não encontrado: ${nomeArquivo}`);
    return;
  }

  try {
    const data = JSON.parse(fs.readFileSync(caminhoAbsoluto, "utf8"));
    const { metadata, weights } = data;

    console.log(`\n=== RELATÓRIO: ${nomeArquivo} ===`);
    console.log(`Geração: ${metadata?.currentCycle || 0}`);
    console.log(`Epsilon: ${metadata?.epsilon || "N/A"}`);

    let totalValores = 0;
    let soma = 0;
    let min = Infinity;
    let max = -Infinity;

    // Agora procurando em layer.data (conforme seu JSON real)
    weights.forEach((layer) => {
      const valores = layer.data || layer.values; // Tenta os dois nomes
      if (valores && Array.isArray(valores)) {
        for (let i = 0; i < valores.length; i++) {
          const v = valores[i];
          soma += v;
          if (v < min) min = v;
          if (v > max) max = v;
          totalValores++;
        }
      }
    });

    if (totalValores === 0) {
      console.log(
        "❌ Continuo sem encontrar pesos. Verifique a estrutura do JSON."
      );
      return;
    }

    const media = soma / totalValores;

    let somaQuadrados = 0;
    weights.forEach((layer) => {
      const valores = layer.data || layer.values;
      if (valores && Array.isArray(valores)) {
        for (let i = 0; i < valores.length; i++) {
          somaQuadrados += Math.pow(valores[i] - media, 2);
        }
      }
    });

    const desvioPadrao = Math.sqrt(somaQuadrados / totalValores);

    console.log(`Total de Pesos: ${totalValores.toLocaleString()}`);
    console.log(`Média: ${media.toFixed(8)}`);
    console.log(`Desvio Padrão (DNA): ${desvioPadrao.toFixed(8)}`);
    console.log(`Range: [${min.toFixed(4)} até ${max.toFixed(4)}]`);

    if (desvioPadrao < 0.00000001) {
      console.log("⚠️ VERDICT: IA ESTAGNADA. Os pesos são idênticos.");
    } else {
      console.log("✅ VERDICT: IA COM DADOS VÁLIDOS.");
    }
  } catch (e) {
    console.log(`❌ Erro no arquivo ${nomeArquivo}: ${e.message}`);
  }
}

analisarIA("brain-data.json");
analisarIA("brain-backup-1770233689277.json");
