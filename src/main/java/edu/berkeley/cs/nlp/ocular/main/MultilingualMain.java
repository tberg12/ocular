package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;
import indexer.Indexer;

import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import threading.BetterThreader;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.data.SplitLineImageLoader;
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
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fig.Option;
import fig.OptionsParser;
import fileio.f;

public class MultilingualMain implements Runnable {

	@Option(gloss = "Path of the directory that contains the input document images.")
	public static String inputPath = null; //"test_img";

	@Option(gloss = "Number of training documents to use.")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Path of the directory that will contain output transcriptions and line extractions.")
	public static String outputPath = null; //"output_dir";

	@Option(gloss = "Path to write the learned language model file to. (Only if learnFont is set to true.)")
	public static String outputLmPath = null; //"lm/cs_trained.lmser";
	
	@Option(gloss = "Path to write the learned font file to. (Only if learnFont is set to true.)")
	public static String outputFontPath = null; //"font/trained.fontser";

	@Option(gloss = "Path to the language model file.")
	public static String initLmPath = null; //"lm/cs_lm.lmser";

	@Option(gloss = "Path of the font initializer file.")
	public static String initFontPath = null; //"font/init.fontser";

	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = false;

	@Option(gloss = "Min horizontal padding between characters in pixels. (Best left at default value: 1.)")
	public static int paddingMinWidth = 1;

	@Option(gloss = "Max horizontal padding between characters in pixels (Best left at default value: 5.)")
	public static int paddingMaxWidth = 5;

	@Option(gloss = "Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)")
	public static boolean markovVerticalOffset = true;

	@Option(gloss = "Size of beam for viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)")
	public static int beamSize = 10;

	@Option(gloss = "Whether to learn the font from the input documents and write the font to a file.")
	public static boolean learnFont = true;

	@Option(gloss = "Number of iterations of EM to use for font learning.")
	public static int numEMIters = 4;

	@Option(gloss = "Engine to use for inner loop of emission cache computation. DEFAULT: Uses Java on CPU, which works on any machine but is the slowest method. OPENCL: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. CUDA: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT;

	@Option(gloss = "GPU ID when using CUDA emission engine.")
	public static int cudaDeviceID = 0;

	@Option(gloss = "Number of threads to use for LFBGS during m-step.")
	public static int numMstepThreads = 8;

	@Option(gloss = "Number of threads to use during emission cache compuation. (Only has affect when emissionEngine is set to DEFAULT.)")
	public static int numEmissionCacheThreads = 8;

	@Option(gloss = "Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)")
	public static int numDecodeThreads = 8;

	@Option(gloss = "Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)")
	public static int decodeBatchSize = 32;

	@Option(gloss = "A language model to be used to assign diacritics to the transcription output.")
	public static boolean allowLanguageSwitchOnPunct = true;

	@Option(gloss = "If there are existing extractions (from ExtractLinesMain), where to find them.")
	public static String existingExtractionsPath = null;

	public static enum EmissionCacheInnerLoopType {
		DEFAULT, OPENCL, CUDA
	};

	public static void main(String[] args) {
		MultilingualMain main = new MultilingualMain();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (inputPath == null) throw new IllegalArgumentException("-inputPath not set");
		if (outputPath == null) throw new IllegalArgumentException("-outputPath not set");
		if (outputLmPath == null) throw new IllegalArgumentException("-outputLmPath not set");
		if (outputFontPath == null) throw new IllegalArgumentException("-outputFontPath not set");
		if (initLmPath == null) throw new IllegalArgumentException("-initLmPath not set");
		if (initFontPath == null) throw new IllegalArgumentException("-initFontPath not set");
		
		long overallNanoTime = System.nanoTime();
		long overallEmissionCacheNanoTime = 0;

		if (!new File(initFontPath).exists()) throw new RuntimeException("initFontPath " + initFontPath + " does not exist");

		File outputDir = new File(outputPath);
		if (!outputDir.exists()) outputDir.mkdirs();

		List<Document> documents = loadDocuments();

		System.out.println("Loading LM..");
		Tuple2<CodeSwitchLanguageModel, SparseTransitionModel> lmAndTransModel = getForwardTransitionModel(initLmPath);
		CodeSwitchLanguageModel lm = lmAndTransModel._1;
		SparseTransitionModel forwardTransitionModel = lmAndTransModel._2;
		Indexer<String> charIndexer = lm.getCharacterIndexer();

		System.out.println("Characters: " + charIndexer.getObjects());
		System.out.println("Num characters: " + charIndexer.size());

		System.out.println("Loading font initializer..");
		Map<String, CharacterTemplate> font = FontInitMain.readFont(initFontPath);
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);

		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();

		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		List<Tuple2<String, Map<String, EvalSuffStats>>> allEvalsPre = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		List<Tuple3<Double, String, String>> allWordPerplexities = new ArrayList<Tuple3<Double, String, String>>();

		if (!learnFont) numEMIters = 1;
		
		List<String> languages = makeList(lm.languages());
		Collections.sort(languages);

		for (int iter = 0; iter < numEMIters; ++iter) {
			System.out.println("Iteration: " + iter + "");

			DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

			Map<String, Integer> languageCounts = new HashMap<String, Integer>();
			for (String l : languages)
				languageCounts.put(l, 1); // add-one smooth

			for (int c = 0; c < templates.length; ++c)
				if (templates[c] != null) templates[c].clearCounts();

			for (Document doc : documents) {
				System.out.println("Document: " + doc.baseName());

				final PixelType[][][] pixels = doc.loadLineImages();

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

					final EmissionModel emissionModel = (markovVerticalOffset ? new CachingEmissionModelExplicitOffset(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop) : new CachingEmissionModel(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop));
					long emissionCacheNanoTime = System.nanoTime();
					emissionModel.rebuildCache();
					overallEmissionCacheNanoTime += (System.nanoTime() - emissionCacheNanoTime);

					long nanoTime = System.nanoTime();
					BeamingSemiMarkovDP dp = new BeamingSemiMarkovDP(emissionModel, forwardTransitionModel, backwardTransitionModel);
					Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeStatesAndWidthsAndJointLogProb = dp.decode(beamSize, numDecodeThreads);
					final TransitionState[][] batchDecodeStates = decodeStatesAndWidthsAndJointLogProb._1._1;
					final int[][] batchDecodeWidths = decodeStatesAndWidthsAndJointLogProb._1._2;
					System.out.println("Decode: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");

					if (iter < numEMIters - 1) {
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

					for (int line = 0; line < emissionModel.numSequences(); ++line) {
						decodeStates[startLine + line] = batchDecodeStates[line];
						decodeWidths[startLine + line] = batchDecodeWidths[line];
					}
				}

				// evaluate

				TransitionState[][] decodeCharStates = new TransitionState[decodeStates.length][];
				for (int i = 0; i < decodeStates.length; ++i) {
					decodeCharStates[i] = new TransitionState[decodeStates[i].length];
					for (int j = 0; j < decodeStates[i].length; ++j)
						decodeCharStates[i][j] = (TransitionState) decodeStates[i][j];
				}

				printTranscription(iter, learnFont, "", doc, allEvals, decodeCharStates, charIndexer, outputPath, lm, allWordPerplexities, languageCounts);
			}

			// m-step

			if (iter < numEMIters - 1) {
				long nanoTime = System.nanoTime();
				{
					final int iterFinal = iter;
					BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
						public void call(Integer c, Object ignore) {
							if (templates[c] != null) templates[c].updateParameters(iterFinal, numEMIters);
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
					for(String language: languages)
						sb.append(language).append("->").append(languageCounts.get(language) / languageCountSum).append("  ");
					System.out.println(sb);
					
					lm = new BasicCodeSwitchLanguageModel(newSubModelsAndPriors, lm.getCharacterIndexer(), lm.getProbKeepSameLanguage(), lm.getMaxOrder());
				}
				System.out.println("New LM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
			}

		}

		if (learnFont) {
			FontInitMain.writeFont(font, outputFontPath);
			CodeSwitchLMTrainMain.writeLM(lm, outputLmPath);
		}

		if (!allEvalsPre.isEmpty()) {
			printEvaluation(allEvals, outputPath + "/eval.pre.txt");
		}
		if (!allEvals.isEmpty()) {
			printEvaluation(allEvals, outputPath + "/eval.txt");
		}

		Collections.sort(allWordPerplexities, new Tuple3.DefaultLexicographicTuple3Comparator<Double, String, String>());
		for (Tuple3<Double, String, String> t : allWordPerplexities)
			System.out.println(t._1 + "\t" + t._2 + "\t" + t._3);

		System.out.println("Emission cache time: " + overallEmissionCacheNanoTime / 1e9 + "s");
		System.out.println("Overall time: " + (System.nanoTime() - overallNanoTime) / 1e9 + "s");
	}

	private Tuple2<CodeSwitchLanguageModel, SparseTransitionModel> getForwardTransitionModel(String lmFilePath) {
		CodeSwitchLanguageModel codeSwitchLM = CodeSwitchLMTrainMain.readLM(lmFilePath);
		System.out.println("Loaded CodeSwitchLanguageModel from " + lmFilePath + "");
		for (String lang : codeSwitchLM.languages()) {
			List<String> chars = new ArrayList<String>();
			for (int i : codeSwitchLM.get(lang).getActiveCharacters())
				chars.add(codeSwitchLM.getCharacterIndexer().getObject(i));
			Collections.sort(chars);
			System.out.println("    " + lang + ": " + chars);
		}

		if (markovVerticalOffset)
			throw new RuntimeException("multilingual markov-vertical-offset transition model not supported");
		else
			return makeTuple2(codeSwitchLM, (SparseTransitionModel) new CodeSwitchTransitionModel(codeSwitchLM, allowLanguageSwitchOnPunct));
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

	private static void printTranscription(int iter, boolean learnFont, String nameAdj, Document doc, List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals, TransitionState[][] decodeStates, Indexer<String> charIndexer, String outputPath, CodeSwitchLanguageModel lm, List<Tuple3<Double, String, String>> allWordPerplexities, Map<String, Integer> languageCounts) {
		final String[][] text = doc.loadLineText();
		int numLines = (text != null ? Math.max(text.length, decodeStates.length) : decodeStates.length); // in case gold and viterbi have different line counts

		// Get the model output
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

		StringBuffer outputBuffer = new StringBuffer();
		if (text != null) {
			// Evaluate against gold-transcribed data (given as "text")
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
				outputBuffer.append(StringHelper.join(viterbiChars[line], "") + "\n");
				outputBuffer.append(StringHelper.join(goldCharSequences[line], "") + "\n");
				outputBuffer.append("\n\n");
			}

			Map<String, EvalSuffStats> evals = Evaluator.getUnsegmentedEval(viterbiChars, goldCharSequences);
			if (iter == MultilingualMain.numEMIters - 1) {
				allEvals.add(makeTuple2(doc.baseName(), evals));
			}
			outputBuffer.append(Evaluator.renderEval(evals));
		}
		else {
			for (int line = 0; line < decodeStates.length; ++line) {
				outputBuffer.append(StringHelper.join(viterbiChars[line], "") + "\n");
			}
		}

		printLanguageAnnotatedTranscription(text, decodeStates, charIndexer, outputBuffer, doc.baseName(), lm, allWordPerplexities, languageCounts);
		System.out.println(outputBuffer.toString());
		f.writeString(outputPath + "/" + doc.baseName().replaceAll("\\.[^.]*$", "") + "." + (learnFont ? "iter-" + iter : "") + nameAdj + ".html", outputBuffer.toString());
	}

	private static void printLanguageAnnotatedTranscription(String[][] text, TransitionState[][] decodeStates, Indexer<String> charIndexer, StringBuffer outputBuffer, String imgFilename, CodeSwitchLanguageModel lm, List<Tuple3<Double, String, String>> allWordPerplexities, Map<String, Integer> languageCounts) {
		outputBuffer.append("\n\n\n\n\n");
		outputBuffer.append("===============================================");
		outputBuffer.append("\n\n\n\n\n");
		{
			List<String>[] csViterbiChars = new List[decodeStates.length];
			String prevLanguage = null;
			for (int line = 0; line < decodeStates.length; ++line) {
				csViterbiChars[line] = new ArrayList<String>();
				if (decodeStates[line] != null) {
					for (int i = 0; i < decodeStates[line].length; ++i) {
						int c = decodeStates[line][i].getCharIndex();
						if (csViterbiChars[line].isEmpty() || !(HYPHEN.equals(csViterbiChars[line].get(csViterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
							String s = charIndexer.getObject(c);
							csViterbiChars[line].add(s);

							String currLanguage = decodeStates[line][i].getLanguage();
							if (!StringHelper.equals(currLanguage, prevLanguage)) {
								if (prevLanguage != null) {
									outputBuffer.append("]");
								}
								outputBuffer.append("[" + currLanguage + " ");
							}
							outputBuffer.append(s);
							prevLanguage = currLanguage;
						}
					}
				}
				outputBuffer.append("\n");
			}
		}
		outputBuffer.append("\n\n\n\n\n");
		{
			outputBuffer.append("<HTML xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
			outputBuffer.append("<HEAD><META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></HEAD>\n");
			outputBuffer.append("<body>\n");
			outputBuffer.append("<table><tr><td>\n");
			outputBuffer.append("<font face=\"courier\"> \n");
			outputBuffer.append("</br></br></br></br></br>\n");
			String[] colors = new String[] { "Black", "Red", "Blue", "Olive", "Orange", "Magenta", "Lime", "Cyan", "Purple", "Green", "Brown" };
			Map<String, String> langColor = new HashMap<String, String>();
			langColor.put(null, colors[0]);
			outputBuffer.append("</br></br>\n\n");

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

			// pl_blac_047_00037-1000.jpg
			// pl_blac_047_00037-1000-line_extract.jpg
			String leImgFilename = FileUtil.withoutExtension(imgFilename) + "-line_extract." + FileUtil.extension(imgFilename);
			outputBuffer.append("</td><td><img src=\"" + leImgFilename + "\">\n");
			outputBuffer.append("</td></tr></table>\n");
			outputBuffer.append("</body></html>\n");
			outputBuffer.append("\n\n\n");
		}
		outputBuffer.append("\n\n\n\n\n");
	}

	private EmissionCacheInnerLoop getEmissionInnerLoop() {
		if (emissionEngine == EmissionCacheInnerLoopType.DEFAULT) return new DefaultInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.OPENCL) return new OpenCLInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.CUDA) return new CUDAInnerLoop(numEmissionCacheThreads, cudaDeviceID);
		throw new RuntimeException("emissionEngine=" + emissionEngine + " not supported");
	}

	private List<Document> loadDocuments() {
		LazyRawImageLoader loader = new LazyRawImageLoader(inputPath, CharacterTemplate.LINE_HEIGHT, binarizeThreshold, crop);
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
		for (int docNum = 0; docNum < Math.min(lazyDocs.size(), numDocs); ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			String baseName = lazyDoc.baseName();
			String preext = FileUtil.withoutExtension(baseName);
			String extension = FileUtil.extension(baseName);

			System.out.println("Loading data for " + inputPath + "/" + baseName);
			if (existingExtractionsPath == null) {
				documents.add(lazyDoc);
				//String fullDocPath = outputPath + "/" + preext + "-line_extract." + extension;
				//final PixelType[][][] pixels = lazyDoc.loadLineImages();
				//System.out.println("Printing line extraction " + fullDocPath);
				//f.writeImage(fullDocPath, Visualizer.renderLineExtraction(pixels));
			}
			else {
				String existingExtractionsDir = existingExtractionsPath + "/line_extract/" + preext + "/";
				System.out.println("existingExtractionsDir is [" + existingExtractionsDir + "], which " + (new File(existingExtractionsDir).exists() ? "exists" : "does not exist"));
				final Pattern pattern = Pattern.compile(preext + "-line_extract-\\d+." + extension);
				File[] lineImageFiles = new File(existingExtractionsDir).listFiles(new FilenameFilter() {
					public boolean accept(File dir, String name) {
						return pattern.matcher(name).matches();
					}
				});
				if (lineImageFiles == null) throw new RuntimeException("lineImageFiles is null");
				if (lineImageFiles.length == 0) throw new RuntimeException("lineImageFiles.length == 0");
				Arrays.sort(lineImageFiles);
				String[] lineImagePaths = new String[lineImageFiles.length];
				for (int i = 0; i < lineImageFiles.length; ++i)
					lineImagePaths[i] = existingExtractionsDir + lineImageFiles[i].getName();
				Document doc = new SplitLineImageLoader.SplitLineImageDocument(lineImagePaths, baseName, CharacterTemplate.LINE_HEIGHT);
				//doc.loadLineImages();
				//doc.loadLineText();
				documents.add(doc);
			}
		}
		return documents;
	}

	private CharacterTemplate[] loadTemplates(Map<String, CharacterTemplate> font, Indexer<String> charIndexer) {
		final CharacterTemplate[] templates = new CharacterTemplate[charIndexer.size()];
		for (int c = 0; c < templates.length; ++c) {
			templates[c] = font.get(charIndexer.getObject(c));
		}
		return templates;
	}

}
