package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicEMDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.BasicEMIterationEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.EMDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.EMIterationEvaluator;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CUDAInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.DefaultInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.FontTrainEM;
import edu.berkeley.cs.nlp.ocular.model.OpenCLInnerLoop;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.NoSubGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fig.Option;
import fig.OptionsParser;
import indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class TranscribeOrTrainFont implements Runnable {

	// Main Options
	
	@Option(gloss = "Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String inputPath = null; //"test_img";

	@Option(gloss = "Path of the directory that will contain output transcriptions.")
	public static String outputPath = null; //"output_dir";

	@Option(gloss = "Path to the input language model file.")
	public static String inputLmPath = null;

	@Option(gloss = "Path of the input font file.")
	public static String inputFontPath = null;

	@Option(gloss = "Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents. Default: use all documents")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.  Default: 0")
	public static int numDocsToSkip = 0;

	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String extractedLinesPath = null;
	
	// Font Learning Options
	
	@Option(gloss = "Whether to learn the font from the input documents and write the font to a file.")
	public static boolean trainFont = false;

	@Option(gloss = "Path to write the learned font file to. (Required if trainFont is set to true, otherwise ignored.)")
	public static String outputFontPath = null;
	
	@Option(gloss = "Number of iterations of EM to use for font learning.  (Only relevant if trainFont is set to true.)  Default: 3")
	public static int numEMIters = 3;
	
	@Option(gloss = "Number of documents to process for each parameter update.  (Only relevant if trainFont is set to true.)  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with trainFont=true and numEMIter=1 (though this could also be achieved by simply running a series of smaller font training jobs each with numEMIter=1, which each subsequent job uses the model output by the previous).  Default is to update only after each full pass over the document set.")
	public static int updateDocBatchSize = Integer.MAX_VALUE;

	@Option(gloss = "Should the counts from each batch accumulate with the previous batches, as opposed to each batch starting fresh?  Note that the counts will always be refreshed after a full pass through the documents.  (Only relevant if trainFont is set to true.)  Default: true")
	public static boolean accumulateBatchesWithinIter = true;
	
	@Option(gloss = "The minimum number of documents that may be used to make a batch for updating parameters.  If the last batch of a pass will contain fewer than this many documents, then lump them in with the last complete batch.  (Only relevant if trainFont is set to true, and updateDocBatchSize is used.)  Default is to always lump remaining documents in with the last complete batch.")
	public static int minDocBatchSize = Integer.MAX_VALUE;

	// Language Model Re-training Options
	
	@Option(gloss = "Should the language model be updated during font training? Default: false")
	public static boolean retrainLM = false;
	
	@Option(gloss = "Path to write the retrained language model file to. (Only relevant if retrainLM is set to true.)  Default: Don't write out the trained LM.")
	public static String outputLmPath = null; //"lm/cs_trained.lmser";

	// Glyph Substitution Model Options
	
	@Option(gloss = "Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions. Default: false")
	public static boolean allowGlyphSubstitution = false;
	
	@Option(gloss = "Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.) Default: Don't use a pre-initialized GSM.")
	public static String inputGsmPath = null;

	@Option(gloss = "Exponent on GSM scores. Default: ")
	public static double gsmPower = 4.0;

	@Option(gloss = "The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions. Default: 0.999999")
	public static double gsmNoCharSubPrior = 0.9999999;
	
	@Option(gloss = "Should the glyph substitution model be updated during font training? (Only relevant if allowGlyphSubstitution is set to true.) Default: false")
	public static boolean retrainGSM = false;
	
	@Option(gloss = "Path to write the retrained glyph substitution model file to. (Only relevant if allowGlyphSubstitution and retrainGSM are set to true.)  Default: Don't write out the trained GSM.")
	public static String outputGsmPath = null;
	
	@Option(gloss = "The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation. Default: 1.0")
	public static double gsmSmoothingCount = 1.0;
	
	@Option(gloss = "Should the GSM consider (condition on) the previous LM char when deciding the glyph to output? (Only relevant if allowGlyphSubstitution is set to true. Default: false")
	public static boolean gsmUsePrevLmChar = false;
	
	@Option(gloss = "A glyph-context combination must be seen at least this many times in the last training iteration if it is to be allowed in the evaluation GSM.  This restricts spurious substitutions during evaluation.  (Only relevant if allowGlyphSubstitution is set to true.)  Default: 2")
	public static int gsmMinCountsForEval = 2;

	// Line Extraction Options
	
	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = true;

	@Option(gloss = "Scale all lines to have the same height?")
	public static boolean uniformLineHeight = true;

	// Miscellaneous Options
	
	@Option(gloss = "Engine to use for inner loop of emission cache computation. DEFAULT: Uses Java on CPU, which works on any machine but is the slowest method. OPENCL: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. CUDA: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT;

	@Option(gloss = "Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)")
	public static int beamSize = 10;

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

	@Option(gloss = "Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)")
	public static boolean markovVerticalOffset = false;

	@Option(gloss = "A language model to be used to assign diacritics to the transcription output.")
	public static boolean allowLanguageSwitchOnPunct = true;
	
	// Options used if evaluation should be performed during training
	
	@Option(gloss = "When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String evalInputPath = null;

	@Option(gloss = "When using -evalInputPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String evalExtractedLinesPath = null;

	@Option(gloss = "When using -evalInputPath, the font trainer will perform an evaluation every `evalFreq` iterations. Default: Evaluate only after all iterations have completed.")
	public static int evalFreq = Integer.MAX_VALUE;
	
	@Option(gloss = "When using -evalInputPath, on iterations in which we run the evaluation, should the evaluation be run after each batch (in addition to after each iteration)?. Default: false")
	public static boolean evalBatches = false;
	
	
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
		if (trainFont && outputFontPath == null) throw new IllegalArgumentException("-outputFontPath required when -trainFont is true.");
		if (!trainFont && outputFontPath != null) throw new IllegalArgumentException("-outputFontPath not permitted when -trainFont is false.");
		if (inputLmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (outputLmPath != null && !retrainLM) throw new IllegalArgumentException("-outputLmPath not permitted if -retrainLM is false.");
		if (retrainGSM && !allowGlyphSubstitution) throw new IllegalArgumentException("-retrainGSM not permitted if -allowGlyphSubstitution is false.");
		if (inputGsmPath != null && !allowGlyphSubstitution) throw new IllegalArgumentException("-inputGsmPath not permitted if -allowGlyphSubstitution is false.");
		if (outputGsmPath != null && !retrainGSM) throw new IllegalArgumentException("-outputGsmPath not permitted if -retrainGsM is false.");
		if (inputFontPath == null) throw new IllegalArgumentException("-inputFontPath not set");
		if (numDocsToSkip < 0) throw new IllegalArgumentException("-numDocsToSkip must be >= 0.  Was "+numDocsToSkip+".");
		if (evalExtractedLinesPath != null && evalInputPath == null) throw new IllegalArgumentException("-evalExtractedLinesPath not permitted without -evalInputPath.");
		
		if (!new File(inputFontPath).exists()) throw new RuntimeException("inputFontPath " + inputFontPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");

		File outputDir = new File(outputPath);
		if (!outputDir.exists()) outputDir.mkdirs();

		List<Document> trainDocuments = loadDocuments(inputPath, extractedLinesPath, numDocs, numDocsToSkip);

		/*
		 * Load LM (and print some info about it)
		 */
		System.out.println("Loading initial LM from " + inputLmPath);
		CodeSwitchLanguageModel codeSwitchLM = TrainLanguageModel.readLM(inputLmPath);
		System.out.println("Loaded CodeSwitchLanguageModel from " + inputLmPath);
		for (int i = 0; i < codeSwitchLM.getLanguageIndexer().size(); ++i) {
			List<String> chars = new ArrayList<String>();
			for (int j : codeSwitchLM.get(i).getActiveCharacters())
				chars.add(codeSwitchLM.getCharacterIndexer().getObject(j));
			Collections.sort(chars);
			System.out.println("    " + codeSwitchLM.getLanguageIndexer().getObject(i) + ": " + chars);
		}

		Indexer<String> charIndexer = codeSwitchLM.getCharacterIndexer();
		Indexer<String> langIndexer = codeSwitchLM.getLanguageIndexer();

		List<String> allCharacters = makeList(charIndexer.getObjects());
		Collections.sort(allCharacters);
		System.out.println("Characters: " + allCharacters);
		System.out.println("Num characters: " + charIndexer.size());

		System.out.println("Loading font initializer from " + inputFontPath);
		Map<String, CharacterTemplate> font = InitializeFont.readFont(inputFontPath);

		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();

		DecoderEM decoderEM = new DecoderEM(emissionInnerLoop, allowGlyphSubstitution, gsmNoCharSubPrior, allowLanguageSwitchOnPunct, markovVerticalOffset, paddingMinWidth, paddingMaxWidth, beamSize, numDecodeThreads, numMstepThreads, decodeBatchSize, charIndexer);
		EMDocumentEvaluator emDocumentEvaluator = new BasicEMDocumentEvaluator(charIndexer, langIndexer, allowGlyphSubstitution);
		
		List<Document> evalDocuments = null;
		EMIterationEvaluator emEvalSetIterationEvaluator;
		if (evalInputPath != null) {
			evalDocuments = loadDocuments(evalInputPath, evalExtractedLinesPath, numDocs, numDocsToSkip);
			emEvalSetIterationEvaluator = new BasicEMIterationEvaluator(evalDocuments, evalInputPath, outputPath, trainFont, numEMIters, decoderEM, emDocumentEvaluator, charIndexer);
		}
		else {
			emEvalSetIterationEvaluator = new EMIterationEvaluator.NoOpEMIterationEvaluator();
		}
			
		/*
		 * Load GSM (and print some info about it)
		 */
		int numLanguages = langIndexer.size();
		@SuppressWarnings("unchecked")
		Set<Integer>[] activeCharacterSets = new Set[numLanguages];
		for (int l = 0; l < numLanguages; ++l) activeCharacterSets[l] = codeSwitchLM.get(l).getActiveCharacters();
		BasicGlyphSubstitutionModelFactory gsmFactory = new BasicGlyphSubstitutionModelFactory(gsmSmoothingCount, langIndexer, charIndexer, activeCharacterSets, !gsmUsePrevLmChar, gsmPower, gsmMinCountsForEval, inputPath, outputPath, trainDocuments, evalDocuments);
		GlyphSubstitutionModel codeSwitchGSM = getGlyphSubstituionModel(gsmFactory, langIndexer, charIndexer);

		FontTrainEM fontTrainEM = new FontTrainEM(langIndexer, charIndexer, decoderEM, gsmFactory, emDocumentEvaluator, accumulateBatchesWithinIter, minDocBatchSize, updateDocBatchSize, numMstepThreads, emEvalSetIterationEvaluator, evalFreq, evalBatches, outputFontPath != null, outputLmPath != null, outputGsmPath != null);
		
		long overallNanoTime = System.nanoTime();
		Tuple3<Map<String, CharacterTemplate>, CodeSwitchLanguageModel, GlyphSubstitutionModel> trainedModels = 
				fontTrainEM.run(trainDocuments, inputPath, outputPath, trainFont, retrainLM, retrainGSM, numEMIters, codeSwitchLM, codeSwitchGSM, font);
		Map<String, CharacterTemplate> newFont = trainedModels._1;
		CodeSwitchLanguageModel newLm = trainedModels._2;
		GlyphSubstitutionModel newGsm = trainedModels._3;
		if (outputFontPath != null) InitializeFont.writeFont(newFont, outputFontPath);
		if (outputLmPath != null) TrainLanguageModel.writeLM(newLm, outputLmPath);
		if (outputGsmPath != null) GlyphSubstitutionModel.writeGSM(newGsm, outputGsmPath);

		System.out.println("Overall time: " + (System.nanoTime() - overallNanoTime) / 1e9 + "s");
	}

	private EmissionCacheInnerLoop getEmissionInnerLoop() {
		if (emissionEngine == EmissionCacheInnerLoopType.DEFAULT) return new DefaultInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.OPENCL) return new OpenCLInnerLoop(numEmissionCacheThreads);
		if (emissionEngine == EmissionCacheInnerLoopType.CUDA) return new CUDAInnerLoop(numEmissionCacheThreads, cudaDeviceID);
		throw new RuntimeException("emissionEngine=" + emissionEngine + " not supported");
	}

	private GlyphSubstitutionModel getGlyphSubstituionModel(BasicGlyphSubstitutionModelFactory gsmFactory, Indexer<String> langIndexer, Indexer<String> charIndexer) {
		if (!allowGlyphSubstitution) {
			System.out.println("Glyph substitution not allowed; constructing no-sub GSM.");
			return new NoSubGlyphSubstitutionModel(langIndexer, charIndexer);
		}
		else if (inputGsmPath != null) { // file path given
			System.out.println("Loading initial GSM from " + inputGsmPath);
			return GlyphSubstitutionModel.readGSM(inputGsmPath);
		}
		else {
			System.out.println("No initial GSM provided; initializing to uniform model.");
			return gsmFactory.uniform();
		}
	}
	
	private List<Document> loadDocuments(String inputPath, String extractedLinesPath, int numDocs, int numDocsToSkip) {
		int lineHeight = uniformLineHeight ? CharacterTemplate.LINE_HEIGHT : -1;
		LazyRawImageLoader loader = new LazyRawImageLoader(inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath);
		List<Document> documents = new ArrayList<Document>();

		List<Document> lazyDocs = loader.readDataset();
		Collections.sort(lazyDocs, new Comparator<Document>() {
			public int compare(Document o1, Document o2) {
				return o1.baseName().compareTo(o2.baseName());
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
		if (actualNumDocsToUse < 1) throw new RuntimeException("No documents given!");
		return documents;
	}
	
}
