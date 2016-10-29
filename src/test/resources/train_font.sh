target/start edu.berkeley.cs.nlp.ocular.main.TrainFont \
  -inputFontPath src/test/resources/doc-init.fontser \
  -inputLmPath src/test/resources/doc.lmser \
  -inputDocPath src/test/resources/doc.jpg \
  -extractedLinesPath src/test/resources/extracted_lines \
  -outputFontPath src/test/resources/doc-trained.fontser \
  -outputPath src/test/resources/train_output \
  -numEmIters 1
#  -allowGlyphSubstitution true \
#  -updateGsm true \
#  -outputGsmPath src/test/resources/doc.gsmser \

target/start edu.berkeley.cs.nlp.ocular.main.TrainFont \
  -inputFontPath src/test/resources/multiling-init.fontser \
  -inputLmPath src/test/resources/multiling.lmser \
  -inputDocPath src/test/resources/doc.jpg \
  -extractedLinesPath src/test/resources/extracted_lines \
  -outputFontPath src/test/resources/multiling-trained.fontser \
  -outputPath src/test/resources/multiling_train_output \
  -numEmIters 1 \
  -allowGlyphSubstitution true \
  -updateGsm true \
  -outputGsmPath src/test/resources/multiling.gsmser \
