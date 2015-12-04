package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CUDAInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DefaultInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.FontTrainEM;
import edu.berkeley.cs.nlp.ocular.model.OpenCLInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.NoSubGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fig.Option;
import fig.OptionsParser;
import fileio.f;
import indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class TranscribeOrTrainFont implements Runnable {

	@Option(gloss = "Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String inputPath = null; //"test_img";

	@Option(gloss = "Number of documents (pages) to use. Ignore or use -1 to use all documents. Default: use all documents")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Number of training documents (pages) to skip over.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.  Default: 0")
	public static int numDocsToSkip = 0;

	@Option(gloss = "Path to the input language model file.")
	public static String lmPath = null; //"lm/cs_lm.lmser";

	@Option(gloss = "Path of the input font file.")
	public static String initFontPath = null; //"font/init.fontser";

	@Option(gloss = "Whether to learn the font from the input documents and write the font to a file.")
	public static boolean learnFont = false;

	@Option(gloss = "Number of iterations of EM to use for font learning.  (Only relevant if learnFont is set to true.)  Default: 3")
	public static int numEMIters = 3;
	
	@Option(gloss = "Number of documents to process for each parameter update.  (Only relevant if learnFont is set to true.)  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with trainFont=true and numEMIter=1 (though this could also be achieved by simply running a series of smaller font training jobs each with numEMIter=1, which each subsequent job uses the model output by the previous).  Default is to update only after each full pass over the document set.")
	public static int updateDocBatchSize = Integer.MAX_VALUE;

	@Option(gloss = "Should the counts from each batch accumulate with the previous batches, as opposed to each batch starting fresh?  Note that the counts will always be refreshed after a full pass through the documents.  (Only relevant if learnFont is set to true.)  Default: true")
	public static boolean accumulateBatchesWithinIter = true;
	
	@Option(gloss = "The minimum number of documents that may be used to make a batch for updating parameters.  If the last batch of a pass will contain fewer than this many documents, then lump them in with the last complete batch.  (Only relevant if learnFont is set to true, and updateDocBatchSize is used.)  Default is to always lump remaining documents in with the last complete batch.")
	public static int minDocBatchSize = Integer.MAX_VALUE;

	@Option(gloss = "Path of the directory that will contain output transcriptions.")
	public static String outputPath = null; //"output_dir";

	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String extractedLinesPath = null;
	
	@Option(gloss = "Path to write the learned font file to. (Required if learnFont is set to true, otherwise ignored.)")
	public static String outputFontPath = null; //"font/trained.fontser";
	
	@Option(gloss = "Should the language model be updated during font training? Default: false")
	public static boolean retrainLM = false;
	
	@Option(gloss = "Path to write the retrained language model file to. (Only relevant if retrainLM is set to true.)  Default: Don't write out the trained LM.")
	public static String outputLmPath = null; //"lm/cs_trained.lmser";

	@Option(gloss = "Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions. Default: false")
	public static boolean allowGlyphSubstitution = false;
	
	@Option(gloss = "Should the glyph substitution model be updated during font training? (Only relevant if allowGlyphSubstitution is set to true.) Default: false")
	public static boolean retrainGSM = false;
	
	@Option(gloss = "Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.) Default: Don't use a pre-initialized GSM.")
	public static String inputGsmPath = null;

	@Option(gloss = "Path to write the retrained glyph substitution model file to. (Only relevant if allowGlyphSubstitution and retrainGSM are set to true.)  Default: Don't write out the trained GSM.")
	public static String outputGsmPath = null;
	
	@Option(gloss = "A language model to be used to assign diacritics to the transcription output.")
	public static boolean allowLanguageSwitchOnPunct = true;

	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = true;

	@Option(gloss = "Scale all lines to have the same height?")
	public static boolean uniformLineHeight = true;

	@Option(gloss = "Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)")
	public static boolean markovVerticalOffset = false;

	@Option(gloss = "Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)")
	public static int beamSize = 10;

	@Option(gloss = "Engine to use for inner loop of emission cache computation. DEFAULT: Uses Java on CPU, which works on any machine but is the slowest method. OPENCL: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. CUDA: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT;

	@Option(gloss = "GPU ID when using CUDA emission engine.")
	public static int cudaDeviceID = 0;

	@Option(gloss = "Number of threads to use for LFBGS during m-step.")
	public static int numMstepThreads = 8;

	@Option(gloss = "Number of threads to use during emission cache compuation. (Only has effect when emissionEngine is set to DEFAULT.)")
	public static int numEmissionCacheThreads = 8;

	@Option(gloss = "Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)")
	public static int numDecodeThreads = 8;

	@Option(gloss = "Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)")
	public static int decodeBatchSize = 32;

	@Option(gloss = "Min horizontal padding between characters in pixels. (Best left at default value: 1.)")
	public static int paddingMinWidth = 1;

	@Option(gloss = "Max horizontal padding between characters in pixels (Best left at default value: 5.)")
	public static int paddingMaxWidth = 5;

	
	public static enum EmissionCacheInnerLoopType { DEFAULT, OPENCL, CUDA };

	
	public static void main(String[] args) {
		TranscribeOrTrainFont main = new TranscribeOrTrainFont();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (inputPath == null) throw new IllegalArgumentException("-inputPath not set");
		if (!new File(inputPath).exists()) throw new IllegalArgumentException("-inputPath "+inputPath+" does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (outputPath == null) throw new IllegalArgumentException("-outputPath not set");
		if (learnFont && outputFontPath == null) throw new IllegalArgumentException("-outputFontPath not set");
		if (lmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (outputLmPath != null && !retrainLM) throw new IllegalArgumentException("-outputLmPath not permitted if -retrainLM is false.");
		if (retrainGSM && !allowGlyphSubstitution) throw new IllegalArgumentException("-retrainGSM not permitted if -allowGlyphSubstitution is false.");
		if (inputGsmPath != null && !allowGlyphSubstitution) throw new IllegalArgumentException("-inputGsmPath not permitted if -allowGlyphSubstitution is false.");
		if (outputGsmPath != null && !retrainGSM) throw new IllegalArgumentException("-outputGsmPath not permitted if -retrainGsM is false.");
		if (initFontPath == null) throw new IllegalArgumentException("-initFontPath not set");
		if (numDocsToSkip < 0) throw new IllegalArgumentException("-numDocsToSkip must be >= 0.  Was "+numDocsToSkip+".");
		
		if (!new File(initFontPath).exists()) throw new RuntimeException("initFontPath " + initFontPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");

		File outputDir = new File(outputPath);
		if (!outputDir.exists()) outputDir.mkdirs();

		List<Document> documents = loadDocuments();

		/*
		 * Load LM (and print some info about it)
		 */
		System.out.println("Loading initial LM from " + lmPath);
		CodeSwitchLanguageModel codeSwitchLM = TrainLanguageModel.readLM(lmPath);
		System.out.println("Loaded CodeSwitchLanguageModel from " + lmPath);
		for (int i = 0; i < codeSwitchLM.getLanguageIndexer().size(); ++i) {
			List<String> chars = new ArrayList<String>();
			for (int j : codeSwitchLM.get(i).getActiveCharacters())
				chars.add(codeSwitchLM.getCharacterIndexer().getObject(j));
			Collections.sort(chars);
			System.out.println("    " + codeSwitchLM.getLanguageIndexer().getObject(i) + ": " + chars);
		}

		Indexer<String> charIndexer = codeSwitchLM.getCharacterIndexer();
		Indexer<String> langIndexer = codeSwitchLM.getLanguageIndexer();

		/*
		 * Load GSM (and print some info about it)
		 */
		BasicGlyphSubstitutionModelFactory gsmFactory = new BasicGlyphSubstitutionModelFactory(langIndexer, charIndexer);
		GlyphSubstitutionModel codeSwitchGSM;
		if (!allowGlyphSubstitution) {
			System.out.println("Glyph substitution not allowed; constructing no-sub GSM.");
			codeSwitchGSM = new NoSubGlyphSubstitutionModel(langIndexer, charIndexer);
		}
		else if (inputGsmPath != null) { // file path given
			System.out.println("Loading initial GSM from " + inputGsmPath);
			codeSwitchGSM = GlyphSubstitutionModel.readGSM(inputGsmPath);
		}
		else {
			System.out.println("No initial GSM provided; initializing to uniform model.");
			codeSwitchGSM = gsmFactory.make(Collections.emptyList(), 0.999999, codeSwitchLM, 0);
		}

		List<String> allCharacters = makeList(charIndexer.getObjects());
		Collections.sort(allCharacters);
		System.out.println("Characters: " + allCharacters);
		System.out.println("Num characters: " + charIndexer.size());

		System.out.println("Loading font initializer from " + initFontPath);
		Map<String, CharacterTemplate> font = InitializeFont.readFont(initFontPath);

		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();

		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();

		EMIterationEvaluator emIterationEvaluator = new BasicEMIterationEvaluator(charIndexer, langIndexer, allEvals);
		FontTrainEM fontTrainEM = new FontTrainEM(langIndexer, charIndexer, retrainLM, retrainGSM, gsmFactory, emissionInnerLoop, emIterationEvaluator, 
				accumulateBatchesWithinIter, minDocBatchSize, updateDocBatchSize, allowGlyphSubstitution, allowLanguageSwitchOnPunct, markovVerticalOffset, 
				paddingMinWidth, paddingMaxWidth, beamSize, numDecodeThreads, numMstepThreads, decodeBatchSize);
		
		long overallNanoTime = System.nanoTime();
		Tuple3<CodeSwitchLanguageModel, GlyphSubstitutionModel, Map<String, CharacterTemplate>> trainedModels = 
				fontTrainEM.run(learnFont, numEMIters, documents, codeSwitchLM, codeSwitchGSM, font);
		CodeSwitchLanguageModel newLm = trainedModels._1;
		GlyphSubstitutionModel newGsm = trainedModels._2;
		Map<String, CharacterTemplate> newFont = trainedModels._3;
		if (learnFont) InitializeFont.writeFont(newFont, outputFontPath);
		if (outputLmPath != null) TrainLanguageModel.writeLM(newLm, outputLmPath);
		if (outputGsmPath != null) GlyphSubstitutionModel.writeGSM(newGsm, outputGsmPath);

		if (!allEvals.isEmpty() && new File(inputPath).isDirectory()) {
			printEvaluation(allEvals, outputPath + "/" + new File(inputPath).getName() + "/eval.txt");
		}

		System.out.println("Overall time: " + (System.nanoTime() - overallNanoTime) / 1e9 + "s");
	}

	private EmissionCacheInnerLoop getEmissionInnerLoop() {
		if (emissionEngine == EmissionCacheInnerLoopType.DEFAULT) return new DefaultInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.OPENCL) return new OpenCLInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.CUDA) return new CUDAInnerLoop(numEmissionCacheThreads, cudaDeviceID);
		throw new RuntimeException("emissionEngine=" + emissionEngine + " not supported");
	}

	private List<Document> loadDocuments() {
		int lineHeight = uniformLineHeight ? CharacterTemplate.LINE_HEIGHT : -1;
		LazyRawImageLoader loader = new LazyRawImageLoader(inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath);
		List<Document> documents = new ArrayList<Document>();

		List<Document> lazyDocs = loader.readDataset();
		Collections.sort(lazyDocs, new Comparator<Document>() {
			public int compare(Document o1, Document o2) {
				return o1.baseName().compareTo(o2.baseName());
			}
			public boolean equals(Object obj) {
				return false;
			}
		});
		
		int actualNumDocsToSkip = Math.min(lazyDocs.size(), numDocsToSkip);
		int actualNumDocsToUse = Math.min(lazyDocs.size() - actualNumDocsToSkip, numDocs <= 0 ? Integer.MAX_VALUE : numDocs);
		System.out.println("Using "+actualNumDocsToUse+" documents (skipping "+actualNumDocsToSkip+")");
		for (int docNum = 0; docNum < actualNumDocsToSkip; ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			System.out.println("  Skipping " + lazyDoc.baseName());
		}
		for (int docNum = actualNumDocsToSkip; docNum < actualNumDocsToSkip+actualNumDocsToUse; ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			System.out.println("  Using " + lazyDoc.baseName());
			documents.add(lazyDoc);
		}
		return documents;
	}
	
	public static interface EMIterationEvaluator {
		public void evaluate(int iter, Document doc, TransitionState[][] decodeStates);
	}

	private class BasicEMIterationEvaluator implements EMIterationEvaluator {
		Indexer<String> charIndexer;
		Indexer<String> langIndexer;
		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals;
		
		public BasicEMIterationEvaluator(Indexer<String> charIndexer, Indexer<String> langIndexer, List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals) {
			this.charIndexer = charIndexer;
			this.langIndexer = langIndexer;
			this.allEvals = allEvals;
		}

		public void evaluate(int iter, Document doc, TransitionState[][] decodeStates) {
			printTranscription(iter, learnFont, doc, allEvals, decodeStates, charIndexer, langIndexer, outputPath);
		}
	}

	public static class NoOpEMIterationEvaluator implements EMIterationEvaluator {
		public void evaluate(int iter, Document doc, TransitionState[][] decodeStates) {}
	}

	public static void printEvaluation(List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals, String outputPath) {
		Map<String, EvalSuffStats> totalSuffStats = new HashMap<String, EvalSuffStats>();
		StringBuffer buf = new StringBuffer();
		buf.append("All evals:\n");
		for (Tuple2<String, Map<String, EvalSuffStats>> docNameAndEvals : allEvals) {
			String docName = docNameAndEvals._1;
			Map<String, EvalSuffStats> evals = docNameAndEvals._2;
			buf.append("Document: " + docName + "\n");
			buf.append(Evaluator.renderEval(evals) + "\n");
			for (String evalType : evals.keySet()) {
				EvalSuffStats eval = evals.get(evalType);
				EvalSuffStats totalEval = totalSuffStats.get(evalType);
				if (totalEval == null) {
					totalEval = new EvalSuffStats();
					totalSuffStats.put(evalType, totalEval);
				}
				totalEval.increment(eval);
			}
		}

		buf.append("\nMarco-avg total eval:\n");
		buf.append(Evaluator.renderEval(totalSuffStats) + "\n");

		f.writeString(outputPath, buf.toString());
		System.out.println("\n" + outputPath);
		System.out.println(buf.toString());
	}

	private static void printTranscription(int iter, boolean learnFont, Document doc, List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals, TransitionState[][] decodeStates, Indexer<String> charIndexer, Indexer<String> langIndexer, String outputPath) {
		final String[][] text = doc.loadLineText();
		int numLines = (text != null ? Math.max(text.length, decodeStates.length) : decodeStates.length); // in case gold and viterbi have different line counts

		// Get the model output
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			if (line < decodeStates.length) {
				for (int i = 0; i < decodeStates[line].length; ++i) {
					GlyphChar glyphChar = decodeStates[line][i].getGlyphChar();
					int c = glyphChar.templateCharIndex;
					if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
						if (!glyphChar.isElided) {
							if (!(i == 0 && c == charIndexer.getIndex(Charset.SPACE))) {
								viterbiChars[line].add(charIndexer.getObject(c));
							}
						}
					}
				}
			}
		}

		String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), new File(doc.baseName()))._2;
		String preext = FileUtil.withoutExtension(new File(doc.baseName()).getName());
		String outputFilenameBase = outputPath + "/" + fileParent + "/" + preext + (learnFont && numEMIters > 1 ? "_iter-" + iter : "");
		String transcriptionOutputFilename = outputFilenameBase + "_transcription.txt";
		String goldComparisonOutputFilename = outputFilenameBase + "_vsGold.txt";
		String htmlOutputFilename = outputFilenameBase + ".html";
		new File(transcriptionOutputFilename).getParentFile().mkdirs();
		
		StringBuffer transcriptionOutputBuffer = new StringBuffer();
		for (int line = 0; line < decodeStates.length; ++line) {
			transcriptionOutputBuffer.append(StringHelper.join(viterbiChars[line], "") + "\n");
		}
		System.out.println("Writing transcription output to " + transcriptionOutputFilename);
		System.out.println(transcriptionOutputBuffer.toString());
		f.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());

		if (text != null) {
			// Evaluate against gold-transcribed data (given as "text")
			StringBuffer goldComparisonOutputBuffer = new StringBuffer();
			goldComparisonOutputBuffer.append("MODEL OUTPUT vs. GOLD TRANSCRIPTION\n\n");
			@SuppressWarnings("unchecked")
			List<String>[] goldCharSequences = new List[numLines];
			for (int line = 0; line < numLines; ++line) {
				goldCharSequences[line] = new ArrayList<String>();
				if (line < text.length) {
					for (int i = 0; i < text[line].length; ++i) {
						goldCharSequences[line].add(text[line][i]);
					}
				}
			}

			for (int line = 0; line < numLines; ++line) {
				goldComparisonOutputBuffer.append(StringHelper.join(viterbiChars[line], "") + "\n");
				goldComparisonOutputBuffer.append(StringHelper.join(goldCharSequences[line], "") + "\n");
				goldComparisonOutputBuffer.append("\n");
			}

			Map<String, EvalSuffStats> evals = Evaluator.getUnsegmentedEval(viterbiChars, goldCharSequences);
			if (!learnFont) {
				allEvals.add(makeTuple2(doc.baseName(), evals));
			}
			goldComparisonOutputBuffer.append(Evaluator.renderEval(evals));

			System.out.println("Writing gold comparison to " + goldComparisonOutputFilename);
			System.out.println(goldComparisonOutputBuffer.toString());
			f.writeString(goldComparisonOutputFilename, goldComparisonOutputBuffer.toString());
		}

		if (langIndexer.size() > 1) {
			System.out.println("Multiple languages being used ("+langIndexer.size()+"), so an html file is being generated to show language switching.");
			System.out.println("Writing html output to " + htmlOutputFilename);
			f.writeString(htmlOutputFilename, printLanguageAnnotatedTranscription(text, decodeStates, charIndexer, langIndexer, doc.baseName(), htmlOutputFilename));
		}
	}

	private static String printLanguageAnnotatedTranscription(String[][] text, TransitionState[][] decodeStates, Indexer<String> charIndexer, Indexer<String> langIndexer, String imgFilename, String htmlOutputFilename) {
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<HTML xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
		outputBuffer.append("<HEAD><META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></HEAD>\n");
		outputBuffer.append("<body>\n");
		outputBuffer.append("<table><tr><td>\n");
		outputBuffer.append("<font face=\"courier\"> \n");
		outputBuffer.append("</br></br></br></br></br>\n");
		outputBuffer.append("</br></br>\n\n");

		String[] colors = new String[] { "Black", "Red", "Blue", "Olive", "Orange", "Magenta", "Lime", "Cyan", "Purple", "Green", "Brown" };

		@SuppressWarnings("unchecked")
		List<String>[] csViterbiChars = new List[decodeStates.length];
		int prevLanguage = -1;
		for (int line = 0; line < decodeStates.length; ++line) {
			csViterbiChars[line] = new ArrayList<String>();
			if (decodeStates[line] != null) {
				for (int i = 0; i < decodeStates[line].length; ++i) {
					TransitionState ts = decodeStates[line][i];
					int lmChar = ts.getLmCharIndex();
					int glyphChar = ts.getGlyphChar().templateCharIndex;
					if (csViterbiChars[line].isEmpty() || !(HYPHEN.equals(csViterbiChars[line].get(csViterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(glyphChar)))) {
						String sglyphChar = Charset.unescapeChar(charIndexer.getObject(glyphChar));
						csViterbiChars[line].add(sglyphChar);

						int currLanguage = ts.getLanguageIndex();
						if (currLanguage != prevLanguage) {
							if (prevLanguage < 0) {
								outputBuffer.append("</font>");
							}
							outputBuffer.append("<font color=\"" + colors[currLanguage+1] + "\">");
						}
						if (lmChar != glyphChar) {
							outputBuffer.append("[" + Charset.unescapeChar(charIndexer.getObject(lmChar)) + "/" + (ts.getGlyphChar().isElided ? "" : sglyphChar) + "]");
						}
						else {
							outputBuffer.append(sglyphChar);
						}
						prevLanguage = currLanguage;
					}
				}
			}
			outputBuffer.append("</br>\n");
		}
		outputBuffer.append("</font></font><br/><br/><br/>\n");
		for (int i = -1; i < langIndexer.size(); ++i) {
			outputBuffer.append("<font color=\"" + colors[i+1] + "\">" + (i < 0 ? "none" : langIndexer.getObject(i)) + "</font></br>\n");
		}

		outputBuffer.append("</td><td><img src=\"" + FileUtil.pathRelativeTo(imgFilename, new File(htmlOutputFilename).getParent()) + "\">\n");
		outputBuffer.append("</td></tr></table>\n");
		outputBuffer.append("</body></html>\n");
		outputBuffer.append("\n\n\n");
		outputBuffer.append("\n\n\n\n\n");
		return outputBuffer.toString();
	}

}
