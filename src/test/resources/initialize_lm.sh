target/start edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel \
  -inputTextPath src/test/resources/doc.txt \
  -outputLmPath src/test/resources/doc.lmser \
  -minCharCount 0

target/start edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel \
  -inputTextPath "Lang1->src/test/resources/doc.txt,Lang2->src/test/resources/doc.txt" \
  -outputLmPath src/test/resources/multiling.lmser \
  -charNgramLength "Lang1->6,Lang2->4" \
  -minCharCount 0

