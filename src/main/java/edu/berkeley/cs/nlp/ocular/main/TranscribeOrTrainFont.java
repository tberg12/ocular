package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import indexer.Indexer;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import threading.BetterThreader;
import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.BeamingSemiMarkovDP;
import edu.berkeley.cs.nlp.ocular.model.CUDAInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.CachingEmissionModel;
import edu.berkeley.cs.nlp.ocular.model.CachingEmissionModelExplicitOffset;
import edu.berkeley.cs.nlp.ocular.model.CharacterNgramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterNgramTransitionModelMarkovOffset;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.CodeSwitchTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.DefaultInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.DenseBigramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.EmissionModel;
import edu.berkeley.cs.nlp.ocular.model.OpenCLInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import fig.Option;
import fig.OptionsParser;
import fileio.f;

public class TranscribeOrTrainFont implements Runnable {

	@Option(gloss = "Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String inputPath = null; //"test_img";

	@Option(gloss = "Number of training documents to use. Ignore or use -1 to use all documents.")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Path to the language model file.")
	public static String lmPath = null; //"lm/cs_lm.lmser";

	@Option(gloss = "Path of the font initializer file.")
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
	
	@Option(gloss = "Path to write the learned language model file to. (Only relevant if learnFont is set to true.)")
	public static String outputLmPath = null; //"lm/cs_trained.lmser";
	
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
		if (initFontPath == null) throw new IllegalArgumentException("-initFontPath not set");
		
		long overallNanoTime = System.nanoTime();
		long overallEmissionCacheNanoTime = 0;

		if (!new File(initFontPath).exists()) throw new RuntimeException("initFontPath " + initFontPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");

		File outputDir = new File(outputPath);
		if (!outputDir.exists()) outputDir.mkdirs();

		List<Document> documents = loadDocuments();
		int numUsableDocs = documents.size();

		System.out.println("Loading initial LM from " + lmPath);
		Tuple2<CodeSwitchLanguageModel, SparseTransitionModel> lmAndTransModel = getForwardTransitionModel(lmPath);
		CodeSwitchLanguageModel lm = lmAndTransModel._1;
		SparseTransitionModel forwardTransitionModel = lmAndTransModel._2;
		Indexer<String> charIndexer = lm.getCharacterIndexer();

		List<String> allCharacters = makeList(charIndexer.getObjects());
		Collections.sort(allCharacters);
		System.out.println("Characters: " + allCharacters);
		System.out.println("Num characters: " + charIndexer.size());

		System.out.println("Loading font initializer from " + initFontPath);
		Map<String, CharacterTemplate> font = InitializeFont.readFont(initFontPath);
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);

		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();

		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();

		if (!learnFont) numEMIters = 0;
		else if (numEMIters <= 0) new RuntimeException("If learnFont=true, then numEMIters must be a positive number.");
		
		List<String> languages = makeList(lm.languages());
		Collections.sort(languages);

		for (int iter = 1; (/* learnFont && */ iter <= numEMIters) || (/* !learnFont && */ iter == 1); ++iter) {
			if (learnFont) System.out.println("Training iteration: " + iter + "  (learnFont=true).");
			else System.out.println("Transcribing (learnFont = false).");

			DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

			// The number of characters assigned to a particular language (to re-estimate language probabilities).
			Map<String, Integer> languageCounts = new HashMap<String, Integer>();
			clearTemplates(templates, languages, languageCounts);

			for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
				Document doc = documents.get(docNum);
				System.out.println("Document: " + doc.baseName());

				final PixelType[][][] pixels = doc.loadLineImages();
				doc.loadLineText();

				// e-step

				TransitionState[][] decodeStates = new TransitionState[pixels.length][0];
				int[][] decodeWidths = new int[pixels.length][0];
				int numBatches = (int) Math.ceil(pixels.length / (double) decodeBatchSize);

				for (int b = 0; b < numBatches; ++b) {
					System.gc();
					System.gc();
					System.gc();

					System.out.println("Batch: " + b);

					int startLine = b * decodeBatchSize;
					int endLine = Math.min((b + 1) * decodeBatchSize, pixels.length);
					PixelType[][][] batchPixels = new PixelType[endLine - startLine][][];
					for (int line = startLine; line < endLine; ++line) {
						batchPixels[line - startLine] = pixels[line];
					}

					final EmissionModel emissionModel = (markovVerticalOffset ? 
							new CachingEmissionModelExplicitOffset(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop) : 
							new CachingEmissionModel(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop));
					long emissionCacheNanoTime = System.nanoTime();
					emissionModel.rebuildCache();
					overallEmissionCacheNanoTime += (System.nanoTime() - emissionCacheNanoTime);

					long nanoTime = System.nanoTime();
					BeamingSemiMarkovDP dp = new BeamingSemiMarkovDP(emissionModel, forwardTransitionModel, backwardTransitionModel);
					Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeStatesAndWidthsAndJointLogProb = dp.decode(beamSize, numDecodeThreads);
					final TransitionState[][] batchDecodeStates = decodeStatesAndWidthsAndJointLogProb._1._1;
					final int[][] batchDecodeWidths = decodeStatesAndWidthsAndJointLogProb._1._2;
					System.out.println("Decode: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
					
					for (int line = 0; line < emissionModel.numSequences(); ++line) {
						decodeStates[startLine + line] = batchDecodeStates[line];
						decodeWidths[startLine + line] = batchDecodeWidths[line];
					}

					if (learnFont) {
						incrementCounts(emissionModel, batchDecodeStates, batchDecodeWidths);
					}
				}

				// evaluate

				TransitionState[][] decodeCharStates = new TransitionState[decodeStates.length][];
				for (int i = 0; i < decodeStates.length; ++i) {
					decodeCharStates[i] = new TransitionState[decodeStates[i].length];
					for (int j = 0; j < decodeStates[i].length; ++j)
						decodeCharStates[i][j] = (TransitionState) decodeStates[i][j];
				}

				printTranscription(iter, learnFont, doc, allEvals, decodeCharStates, charIndexer, outputPath, lm, languageCounts);
				
				// m-step
				if (learnFont) {
					int trueMinBatchSize = Math.min(minDocBatchSize, updateDocBatchSize); // min batch size may not exceed standard batch size
					if (docNum+1 == numUsableDocs) { // last document of the set
						lm = updateParameters(iter, templates, lm, languages, languageCounts);
						if (!accumulateBatchesWithinIter) clearTemplates(templates, languages, languageCounts);
					}
					else if (numUsableDocs - (docNum+1) < trueMinBatchSize) { // next batch will be too small, so lump the remaining documents in with this one
						// no update
					} 
					else if ((docNum+1) % updateDocBatchSize == 0) { // batch is complete
						lm = updateParameters(iter, templates, lm, languages, languageCounts);
						if (!accumulateBatchesWithinIter) clearTemplates(templates, languages, languageCounts);
					}
				}
			}

		}

		if (learnFont) {
			InitializeFont.writeFont(font, outputFontPath);
			
			if (outputLmPath != null) {
				TrainLanguageModel.writeLM(lm, outputLmPath);
			}
		}

		if (!allEvals.isEmpty() && new File(inputPath).isDirectory()) {
			printEvaluation(allEvals, outputPath + "/" + new File(inputPath).getName() + "/eval.txt");
		}

		System.out.println("Emission cache time: " + overallEmissionCacheNanoTime / 1e9 + "s");
		System.out.println("Overall time: " + (System.nanoTime() - overallNanoTime) / 1e9 + "s");
	}

	private void clearTemplates(final CharacterTemplate[] templates, List<String> languages, Map<String, Integer> languageCounts) {
		for (int c = 0; c < templates.length; ++c) {
			if (templates[c] != null) templates[c].clearCounts();
		}
		for (String l : languages) languageCounts.put(l, 1); // add-one smooth
	}

	private void incrementCounts(final EmissionModel emissionModel, final TransitionState[][] batchDecodeStates, final int[][] batchDecodeWidths) {
		long nanoTime;
		nanoTime = System.nanoTime();
		BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
			public void call(Integer line, Object ignore) {
				emissionModel.incrementCounts(line, batchDecodeStates[line], batchDecodeWidths[line]);
			}
		};
		BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
		for (int line = 0; line < emissionModel.numSequences(); ++line)
			threader.addFunctionArgument(line);
		threader.run();
		System.out.println("Increment counts: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
	}

	private CodeSwitchLanguageModel updateParameters(int iter, final CharacterTemplate[] templates, CodeSwitchLanguageModel lm, List<String> languages, Map<String, Integer> languageCounts) {
		System.out.println();
		long nanoTime = System.nanoTime();
		{
			final int iterFinal = iter;
			BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
				public void call(Integer c, Object ignore) {
					if (templates[c] != null) templates[c].updateParameters(iterFinal);
				}
			};
			BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
			for (int c = 0; c < templates.length; ++c)
				threader.addFunctionArgument(c);
			threader.run();
		}
		System.out.println("Update parameters: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");

		//
		// Hard-EM update on language probabilities
		//
		nanoTime = System.nanoTime();
		{
			Map<String, Tuple2<SingleLanguageModel, Double>> newSubModelsAndPriors = new HashMap<String, Tuple2<SingleLanguageModel, Double>>();
			double languageCountSum = 0;
			for (String language: languages) {
				double newPrior = languageCounts.get(language).doubleValue();
				newSubModelsAndPriors.put(language, makeTuple2(lm.get(language), newPrior));
				languageCountSum += newPrior;
			}

			StringBuilder sb = new StringBuilder("Updating language probabilities: ");
			for (String language: languages)
				sb.append(language).append("->").append(languageCounts.get(language) / languageCountSum).append("  ");
			System.out.println(sb);
			
			lm = new BasicCodeSwitchLanguageModel(newSubModelsAndPriors, lm.getCharacterIndexer(), lm.getProbKeepSameLanguage(), lm.getMaxOrder());
		}
		System.out.println("New LM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		System.out.println();
		return lm;
	}

	private Tuple2<CodeSwitchLanguageModel, SparseTransitionModel> getForwardTransitionModel(String lmFilePath) {
		CodeSwitchLanguageModel codeSwitchLM = TrainLanguageModel.readLM(lmFilePath);
		System.out.println("Loaded CodeSwitchLanguageModel from " + lmFilePath);
		for (String lang : codeSwitchLM.languages()) {
			List<String> chars = new ArrayList<String>();
			for (int i : codeSwitchLM.get(lang).getActiveCharacters())
				chars.add(codeSwitchLM.getCharacterIndexer().getObject(i));
			Collections.sort(chars);
			System.out.println("    " + lang + ": " + chars);
		}

		CodeSwitchLanguageModel lm;
		SparseTransitionModel transitionModel;
		if (codeSwitchLM.languages().size() > 1) {
			lm = codeSwitchLM; 
			if (markovVerticalOffset)
				throw new RuntimeException("Markov vertical offset transition model not currently supported for multiple languages.");
			else { 
				transitionModel = new CodeSwitchTransitionModel(codeSwitchLM, allowLanguageSwitchOnPunct);
				System.out.println("Using CodeSwitchLanguageModel and CodeSwitchTransitionModel");
			}
		}
		else { // only one language, default to original (monolingual) Ocular code because it will be faster.
			String onlyLanguage = codeSwitchLM.languages().iterator().next();
			SingleLanguageModel singleLm = codeSwitchLM.get(onlyLanguage);
			lm = new OnlyOneLanguageCodeSwitchLM(onlyLanguage, singleLm);
			if (markovVerticalOffset) {
				transitionModel = new CharacterNgramTransitionModelMarkovOffset(singleLm, singleLm.getMaxOrder());
				System.out.println("Using OnlyOneLanguageCodeSwitchLM and CharacterNgramTransitionModelMarkovOffset");
			} else {
				transitionModel = new CharacterNgramTransitionModel(singleLm, singleLm.getMaxOrder());
				System.out.println("Using OnlyOneLanguageCodeSwitchLM and CharacterNgramTransitionModel");
			}
		}
		return makeTuple2(lm, transitionModel); 
	}
	
	private class OnlyOneLanguageCodeSwitchLM implements CodeSwitchLanguageModel, SingleLanguageModel {
		private static final long serialVersionUID = 3287238927893L;
		
		private String language;
		private SingleLanguageModel singleLm;
		public OnlyOneLanguageCodeSwitchLM(String language, SingleLanguageModel singleLm) {
			this.language = language;
			this.singleLm = singleLm;
		}

		public double getCharNgramProb(int[] context, int c) { return singleLm.getCharNgramProb(context, c); }
		public Indexer<String> getCharacterIndexer() { return singleLm.getCharacterIndexer(); }
		public int getMaxOrder() { return singleLm.getMaxOrder(); }
		public Set<Integer> getActiveCharacters() { return singleLm.getActiveCharacters(); }
		public boolean containsContext(int[] context) { return singleLm.containsContext(context); }
		public Set<String> languages() { return Collections.singleton(language); }
		public double getProbKeepSameLanguage() { return 1.0; }
		public Double languagePrior(String language) { 
			return language.equals(this.language) ? 1.0 : 0.0;
    }
		public Double languageTransitionPrior(String fromLanguage, String destinationLanguage) { 
			return fromLanguage.equals(this.language) && destinationLanguage.equals(this.language) ? 1.0 : 0.0; 
		}
		public SingleLanguageModel get(String language) { 
			if (language.equals(this.language)) return singleLm; 
			else throw new RuntimeException("No model found for language '"+language+"'.  (Only found '"+this.language+"')"); 
		}
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

	private static void printTranscription(int iter, boolean learnFont, Document doc, List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals, TransitionState[][] decodeStates, Indexer<String> charIndexer, String outputPath, CodeSwitchLanguageModel lm, Map<String, Integer> languageCounts) {
		final String[][] text = doc.loadLineText();
		int numLines = (text != null ? Math.max(text.length, decodeStates.length) : decodeStates.length); // in case gold and viterbi have different line counts

		// Get the model output
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			if (line < decodeStates.length) {
				for (int i = 0; i < decodeStates[line].length; ++i) {
					int c = decodeStates[line][i].getCharIndex();
					if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
						viterbiChars[line].add(charIndexer.getObject(c));
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
				goldComparisonOutputBuffer.append("\n\n");
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

		if (lm.languages().size() > 1) {
			System.out.println("Multiple languages being used ("+lm.languages().size()+"), so an html file is being generated to show language switching.");
			System.out.println("Writing html output to " + htmlOutputFilename);
			f.writeString(htmlOutputFilename, printLanguageAnnotatedTranscription(text, decodeStates, charIndexer, doc.baseName(), htmlOutputFilename, lm, languageCounts));
		}
	}

	private static String printLanguageAnnotatedTranscription(String[][] text, TransitionState[][] decodeStates, Indexer<String> charIndexer, String imgFilename, String htmlOutputFilename, CodeSwitchLanguageModel lm, Map<String, Integer> languageCounts) {
		StringBuffer outputBuffer = new StringBuffer();

		//		{
		//			List<String>[] csViterbiChars = new List[decodeStates.length];
		//			String prevLanguage = null;
		//			for (int line = 0; line < decodeStates.length; ++line) {
		//				csViterbiChars[line] = new ArrayList<String>();
		//				if (decodeStates[line] != null) {
		//					for (int i = 0; i < decodeStates[line].length; ++i) {
		//						int c = decodeStates[line][i].getCharIndex();
		//						if (csViterbiChars[line].isEmpty() || !(HYPHEN.equals(csViterbiChars[line].get(csViterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
		//							String s = charIndexer.getObject(c);
		//							csViterbiChars[line].add(s);
		//
		//							String currLanguage = decodeStates[line][i].getLanguage();
		//							if (!StringHelper.equals(currLanguage, prevLanguage)) {
		//								if (prevLanguage != null) {
		//									outputBuffer.append("]");
		//								}
		//								outputBuffer.append("[" + currLanguage + " ");
		//							}
		//							outputBuffer.append(s);
		//							prevLanguage = currLanguage;
		//						}
		//					}
		//				}
		//				outputBuffer.append("\n");
		//			}
		//		}
		//		outputBuffer.append("\n\n\n\n\n");
		{
			outputBuffer.append("<HTML xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
			outputBuffer.append("<HEAD><META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></HEAD>\n");
			outputBuffer.append("<body>\n");
			outputBuffer.append("<table><tr><td>\n");
			outputBuffer.append("<font face=\"courier\"> \n");
			outputBuffer.append("</br></br></br></br></br>\n");
			outputBuffer.append("</br></br>\n\n");

			String[] colors = new String[] { "Black", "Red", "Blue", "Olive", "Orange", "Magenta", "Lime", "Cyan", "Purple", "Green", "Brown" };
			Map<String, String> langColor = new HashMap<String, String>();
			langColor.put(null, colors[0]);
			List<String> allLanguages = makeList(lm.languages());
			Collections.sort(allLanguages);
			for (String language: allLanguages) {
				langColor.put(language, colors[langColor.size()]);
			}

			@SuppressWarnings("unchecked")
			List<String>[] csViterbiChars = new List[decodeStates.length];
			String prevLanguage = null;
			for (int line = 0; line < decodeStates.length; ++line) {
				csViterbiChars[line] = new ArrayList<String>();
				if (decodeStates[line] != null) {
					for (int i = 0; i < decodeStates[line].length; ++i) {
						int c = decodeStates[line][i].getCharIndex();
						if (csViterbiChars[line].isEmpty() || !(HYPHEN.equals(csViterbiChars[line].get(csViterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
							String s = Charset.unescapeChar(charIndexer.getObject(c));
							csViterbiChars[line].add(s);

							String currLanguage = decodeStates[line][i].getLanguage();
							if (!StringHelper.equals(currLanguage, prevLanguage)) {
								if (prevLanguage != null) {
									outputBuffer.append("</font>");
								}
								if (!langColor.containsKey(currLanguage)) langColor.put(currLanguage, colors[langColor.size()]);
								outputBuffer.append("<font color=\"" + langColor.get(currLanguage) + "\">");
							}
							if (currLanguage != null) languageCounts.put(currLanguage, languageCounts.get(currLanguage) + 1);
							outputBuffer.append(s);
							prevLanguage = currLanguage;
						}
					}
				}
				outputBuffer.append("</br>\n");
			}
			outputBuffer.append("</font></font><br/><br/><br/>\n");
			for (Map.Entry<String, String> c : langColor.entrySet()) {
				outputBuffer.append("<font color=\"" + c.getValue() + "\">" + c.getKey() + "</font></br>\n");
			}

			outputBuffer.append("</td><td><img src=\"" + FileUtil.pathRelativeTo(imgFilename, new File(htmlOutputFilename).getParent()) + "\">\n");
			outputBuffer.append("</td></tr></table>\n");
			outputBuffer.append("</body></html>\n");
			outputBuffer.append("\n\n\n");
		}
		outputBuffer.append("\n\n\n\n\n");
		return outputBuffer.toString();
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
		
		int numDocsToUse = Math.min(lazyDocs.size(), numDocs <= 0 ? Integer.MAX_VALUE : numDocs);
		System.out.println("Using " + numDocsToUse + " documents");
		for (int docNum = 0; docNum < numDocsToUse; ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			System.out.println("  Using " + lazyDoc.baseName());
			documents.add(lazyDoc);
		}
		return documents;
	}

	private CharacterTemplate[] loadTemplates(Map<String, CharacterTemplate> font, Indexer<String> charIndexer) {
		final CharacterTemplate[] templates = new CharacterTemplate[charIndexer.size()];
		for (int c = 0; c < charIndexer.size(); ++c) {
			CharacterTemplate template = font.get(charIndexer.getObject(c));
			if (template == null)
				throw new RuntimeException("No template found for character '"+charIndexer.getObject(c)+"' ("+StringHelper.toUnicode(charIndexer.getObject(c))+")");
			templates[c] = template;
		}
		return templates;
	}

}
