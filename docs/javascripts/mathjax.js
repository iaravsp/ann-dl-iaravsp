window.MathJax = {
  // \boldsymbol nao faz parte do pacote base do MathJax 3; sem carregar a
  // extensao, o macro aparece em vermelho na pagina.
  loader: {
    load: ["[tex]/boldsymbol"]
  },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  // Com navigation.instant, este callback pode rodar antes de o MathJax
  // terminar de inicializar. Sem a guarda, MathJax.startup.output e null,
  // a chamada lanca TypeError e o typeset nunca acontece: as formulas da
  // pagina ficam sem renderizar.
  if (typeof MathJax === "undefined" || !MathJax.startup) {
    return;
  }

  MathJax.startup.promise = MathJax.startup.promise.then(() => {
    if (MathJax.startup.output) {
      MathJax.startup.output.clearCache();
    }
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
