package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.BasicMultiDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CUDAInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.CachingEmissionModel.CachingEmissionModelFactory;
import edu.berkeley.cs.nlp.ocular.model.CachingEmissionModelExplicitOffset.CachingEmissionModelExplicitOffsetFactory;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.DefaultInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.EmissionModel.EmissionModelFactory;
import edu.berkeley.cs.nlp.ocular.model.FontTrainEM;
import edu.berkeley.cs.nlp.ocular.model.OpenCLInnerLoop;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModelReadWrite;
import edu.berkeley.cs.nlp.ocular.sub.NoSubGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fig.Option;
import fig.OptionsParser;
import indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TranscribeOrTrainFont implements Runnable {

	// ##### Main Options
	
	@Option(gloss = "Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String inputDocPath = null; // Required.

	@Option(gloss = "Path of the directory that will contain output transcriptions.")
	public static String outputPath = null; // Required.

	@Option(gloss = "Path to the input language model file.")
	public static String inputLmPath = null; // Required.

	@Option(gloss = "Path of the input font file.")
	public static String inputFontPath = null; // Required.

	@Option(gloss = "Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents. Default: Use all documents.")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.")
	public static int numDocsToSkip = 0;

	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String extractedLinesPath = null; // Don't read or write line image files.
	
	// ##### Font Learning Options
	
	@Option(gloss = "Whether to learn the font from the input documents and write the font to a file.")
	public static boolean trainFont = false;
	
	// The following options are only relevant if trainFont is set to "true".

	@Option(gloss = "Path to write the learned font file to. Required if trainFont is set to true, otherwise ignored.")
	public static String outputFontPath = null;
	
	@Option(gloss = "Number of iterations of EM to use for font learning.")
	public static int numEMIters = 3;
	
	@Option(gloss = "Number of documents to process for each parameter update.  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with trainFont=true and numEMIter=1 (though this could also be achieved by simply running a series of smaller font training jobs each with numEMIter=1, which each subsequent job uses the model output by the previous).  Default: Update only after each full pass over the document set.")
	public static int updateDocBatchSize = Integer.MAX_VALUE;

	@Option(gloss = "Should the counts from each batch accumulate with the previous batches, as opposed to each batch starting fresh?  Note that the counts will always be refreshed after a full pass through the documents.")
	public static boolean accumulateBatchesWithinIter = false;
	
	@Option(gloss = "The minimum number of documents that may be used to make a batch for updating parameters.  If the last batch of a pass will contain fewer than this many documents, then lump them in with the last complete batch.  Default: Always lump remaining documents of an incomplete batch in with the last complete batch.")
	public static int minDocBatchSize = Integer.MAX_VALUE;
	
	@Option(gloss = "If true, the font trainer will find the latest completed iteration in the outputPath and load it in order to pick up training from that point.  Convenient if a training run crashes when only partially completed.")
	public static boolean continueFromLastCompleteIteration = false;

	// ##### Language Model Re-training Options
	
	@Option(gloss = "Should the language model be updated during font training?")
	public static boolean retrainLM = false;
	
	@Option(gloss = "Path to write the retrained language model file to. (Only relevant if retrainLM is set to true.)  Default: Don't write out the trained LM.")
	public static String outputLmPath = null;

	// ##### Glyph Substitution Model Options
	
	@Option(gloss = "Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions.")
	public static boolean allowGlyphSubstitution = false;
	
	// The following options are only relevant if allowGlyphSubstitution is set to "true".
	
	@Option(gloss = "Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.) Default: Don't use a pre-initialized GSM. (Learn one from scratch).")
	public static String inputGsmPath = null;

	@Option(gloss = "Exponent on GSM scores.")
	public static double gsmPower = 4.0;

	@Option(gloss = "The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions.")
	public static double gsmNoCharSubPrior = 0.9;

	@Option(gloss = "Should the GSM be allowed to elide letters even without the presence of an elision-marking tilde?")
	public static boolean gsmElideAnything = false;
	
	@Option(gloss = "Should the glyph substitution model be updated during font training? (Only relevant if allowGlyphSubstitution is set to true.)")
	public static boolean trainGsm = false;
	
	@Option(gloss = "Path to write the retrained glyph substitution model file to. (Only relevant if allowGlyphSubstitution and trainGsm are set to true.)  Default: Don't write out the trained GSM.")
	public static String outputGsmPath = null;
	
	@Option(gloss = "The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.")
	public static double gsmSmoothingCount = 1.0;
	
	@Option(gloss = "gsmElisionSmoothingCountMultiplier.")
	public static double gsmElisionSmoothingCountMultiplier = 100.0;
	
	@Option(gloss = "A glyph-context combination must be seen at least this many times in the last training iteration if it is to be allowed in the evaluation GSM.  This restricts spurious substitutions during evaluation.  (Only relevant if allowGlyphSubstitution is set to true.)")
	public static int gsmMinCountsForEval = 2;
	
	// ##### Line Extraction Options
	
	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = true;

	@Option(gloss = "Scale all lines to have the same height?")
	public static boolean uniformLineHeight = true;

	// ##### Miscellaneous Options
	
	@Option(gloss = "Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT; // Default: DEFAULT

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

	@Option(gloss = "Min horizontal padding between characters in pixels. (Best left at default value.)")
	public static int paddingMinWidth = 1;

	@Option(gloss = "Max horizontal padding between characters in pixels (Best left at default value.)")
	public static int paddingMaxWidth = 5;

	@Option(gloss = "Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)")
	public static boolean markovVerticalOffset = false;

	@Option(gloss = "A language model to be used to assign diacritics to the transcription output.")
	public static boolean allowLanguageSwitchOnPunct = true;
	
	// ##### Options used if evaluation should be performed during training
	
	@Option(gloss = "When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String evalInputDocPath = null; // Do not evaluate during font training.

	// The following options are only relevant if a value was given to -evalInputDocPath.
	
	@Option(gloss = "When using -evalInputDocPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String evalExtractedLinesPath = null; // Don't read or write line image files. 

	@Option(gloss = "When using -evalInputDocPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents. Default: Use all documents in the specified path.")
	public static int evalNumDocs = Integer.MAX_VALUE;

	@Option(gloss = "When using -evalInputDocPath, the font trainer will perform an evaluation every `evalFreq` iterations. Default: Evaluate only after all iterations have completed.")
	public static int evalFreq = Integer.MAX_VALUE; 
	
	@Option(gloss = "When using -evalInputDocPath, on iterations in which we run the evaluation, should the evaluation be run after each batch (in addition to after each iteration)?")
	public static boolean evalBatches = false;
	
	
	public static enum EmissionCacheInnerLoopType { DEFAULT, OPENCL, CUDA };

	
	public static void main(String[] args) {
		TranscribeOrTrainFont main = new TranscribeOrTrainFont();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		validateOptions();
		main.run();
	}

	public void run() {
		CodeSwitchLanguageModel lm = loadLM();
		Map<String, CharacterTemplate> font = loadFont();
		BasicGlyphSubstitutionModelFactory gsmFactory = makeGsmFactory(lm);
		GlyphSubstitutionModel gsm = getGlyphSubstituionModel(gsmFactory);
		
		Indexer<String> charIndexer = lm.getCharacterIndexer();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		
		DecoderEM decoderEM = makeDecoder(charIndexer);

		boolean evalCharIncludesDiacritic = true;
		SingleDocumentEvaluator documentEvaluator = new BasicSingleDocumentEvaluator(charIndexer, langIndexer, allowGlyphSubstitution, evalCharIncludesDiacritic);
		
		List<Document> documents = LazyRawImageLoader.loadDocuments(inputDocPath, extractedLinesPath, numDocs, numDocsToSkip, false, uniformLineHeight, binarizeThreshold, crop);
		if (trainFont) {
			MultiDocumentEvaluator evalSetEvaluator = makeEvalSetEvaluator(charIndexer, decoderEM, documentEvaluator);
			train(documents, lm, font, gsmFactory, gsm, decoderEM, documentEvaluator, evalSetEvaluator);
		}
		else { /* transcribe only */
			MultiDocumentEvaluator evalSetEvaluator = new BasicMultiDocumentEvaluator(documents, inputDocPath, outputPath, decoderEM, documentEvaluator, charIndexer);
			System.out.println("Transcribing input data      " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			evalSetEvaluator.printTranscriptionWithEvaluation(0, 0, lm, gsm, font);
		}
	}

	private static void validateOptions() {
		if (inputDocPath == null) throw new IllegalArgumentException("-inputDocPath not set");
		if (!new File(inputDocPath).exists()) throw new IllegalArgumentException("-inputDocPath "+inputDocPath+" does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (outputPath == null) throw new IllegalArgumentException("-outputPath not set");
		if (trainFont && numEMIters <= 0) new IllegalArgumentException("-numEMIters must be a positive number if -trainFont is true.");
		if (trainFont && outputFontPath == null) throw new IllegalArgumentException("-outputFontPath required when -trainFont is true.");
		if (!trainFont && outputFontPath != null) throw new IllegalArgumentException("-outputFontPath not permitted when -trainFont is false.");
		if (inputLmPath == null) throw new IllegalArgumentException("-lmPath not set");
		if (outputLmPath != null && !retrainLM) throw new IllegalArgumentException("-outputLmPath not permitted if -retrainLM is false.");
		if (trainGsm && !allowGlyphSubstitution) throw new IllegalArgumentException("-trainGsm not permitted if -allowGlyphSubstitution is false.");
		if (inputGsmPath != null && !allowGlyphSubstitution) throw new IllegalArgumentException("-inputGsmPath not permitted if -allowGlyphSubstitution is false.");
		if (outputGsmPath != null && !trainGsm) throw new IllegalArgumentException("-outputGsmPath not permitted if -trainGsm is false.");
		if (inputFontPath == null) throw new IllegalArgumentException("-inputFontPath not set");
		if (numDocsToSkip < 0) throw new IllegalArgumentException("-numDocsToSkip must be >= 0.  Was "+numDocsToSkip+".");
		if (evalExtractedLinesPath != null && evalInputDocPath == null) throw new IllegalArgumentException("-evalExtractedLinesPath not permitted without -evalInputDocPath.");
		if (!new File(inputFontPath).exists()) throw new RuntimeException("inputFontPath " + inputFontPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");

		// Make the output directory if it doesn't exist yet
		File outputPathFile = new File(outputPath);
		if (!outputPathFile.exists()) outputPathFile.mkdirs();
	}

	private void train(List<Document> trainDocuments, CodeSwitchLanguageModel lm, Map<String, CharacterTemplate> font,
			BasicGlyphSubstitutionModelFactory gsmFactory, GlyphSubstitutionModel gsm, DecoderEM decoderEM,
			SingleDocumentEvaluator documentEvaluator, MultiDocumentEvaluator evalSetIterationEvaluator) {
		Tuple3<Map<String, CharacterTemplate>, CodeSwitchLanguageModel, GlyphSubstitutionModel> trainedModels = 
				new FontTrainEM().train(
						trainDocuments, 
						lm, gsm, font, 
						retrainLM, trainGsm, 
						continueFromLastCompleteIteration,
						outputFontPath != null, outputLmPath != null, outputGsmPath != null,
						decoderEM,
						gsmFactory, documentEvaluator,
						numEMIters, updateDocBatchSize, minDocBatchSize, accumulateBatchesWithinIter,
						numMstepThreads,
						inputDocPath, outputPath,
						evalSetIterationEvaluator, evalFreq, evalBatches);
		
		Map<String, CharacterTemplate> newFont = trainedModels._1;
		CodeSwitchLanguageModel newLm = trainedModels._2;
		GlyphSubstitutionModel newGsm = trainedModels._3;
		if (outputFontPath != null) InitializeFont.writeFont(newFont, outputFontPath);
		if (outputLmPath != null) TrainLanguageModel.writeLM(newLm, outputLmPath);
		if (outputGsmPath != null) GlyphSubstitutionModelReadWrite.writeGSM(newGsm, outputGsmPath);
	}

	private Map<String, CharacterTemplate> loadFont() {
		System.out.println("Loading font from " + inputFontPath);
		Map<String, CharacterTemplate> font = InitializeFont.readFont(inputFontPath);
		return font;
	}
	
	private CodeSwitchLanguageModel loadLM() {
		System.out.println("Loading initial LM from " + inputLmPath);
		CodeSwitchLanguageModel codeSwitchLM = TrainLanguageModel.readLM(inputLmPath);

		//print some useful info
		System.out.println("Loaded CodeSwitchLanguageModel from " + inputLmPath);
		Indexer<String> charIndexer = codeSwitchLM.getCharacterIndexer();
		for (int i = 0; i < codeSwitchLM.getLanguageIndexer().size(); ++i) {
			List<String> chars = new ArrayList<String>();
			for (int j : codeSwitchLM.get(i).getActiveCharacters())
				chars.add(charIndexer.getObject(j));
			Collections.sort(chars);
			System.out.println("    " + codeSwitchLM.getLanguageIndexer().getObject(i) + ": " + chars);
		}
		List<String> allCharacters = makeList(charIndexer.getObjects());
		Collections.sort(allCharacters);
		System.out.println("Characters: " + allCharacters);
		System.out.println("Num characters: " + charIndexer.size());

		return codeSwitchLM;
	}

	private BasicGlyphSubstitutionModelFactory makeGsmFactory(CodeSwitchLanguageModel lm) {
		Indexer<String> charIndexer = lm.getCharacterIndexer();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		int numLanguages = langIndexer.size();
		@SuppressWarnings("unchecked")
		Set<Integer>[] activeCharacterSets = new Set[numLanguages];
		for (int l = 0; l < numLanguages; ++l) activeCharacterSets[l] = lm.get(l).getActiveCharacters();
		return new BasicGlyphSubstitutionModelFactory(gsmSmoothingCount, gsmElisionSmoothingCountMultiplier, langIndexer, charIndexer, activeCharacterSets, gsmPower, gsmMinCountsForEval, outputPath);
	}

	private GlyphSubstitutionModel getGlyphSubstituionModel(BasicGlyphSubstitutionModelFactory gsmFactory) {
		if (!allowGlyphSubstitution) {
			System.out.println("Glyph substitution not allowed; constructing no-sub GSM.");
			return new NoSubGlyphSubstitutionModel();
		}
		else if (inputGsmPath != null) { // file path given
			System.out.println("Loading initial GSM from " + inputGsmPath);
			return GlyphSubstitutionModelReadWrite.readGSM(inputGsmPath);
		}
		else {
			System.out.println("No initial GSM provided; initializing to uniform model.");
			return gsmFactory.uniform();
		}
	}
	
	private DecoderEM makeDecoder(Indexer<String> charIndexer) {
		EmissionModelFactory emissionModelFactory = makeEmissionModelFactory(charIndexer);
		return new DecoderEM(emissionModelFactory, allowGlyphSubstitution, gsmNoCharSubPrior, gsmElideAnything, allowLanguageSwitchOnPunct, markovVerticalOffset, beamSize, numDecodeThreads, numMstepThreads, decodeBatchSize);
	}

	private EmissionModelFactory makeEmissionModelFactory(Indexer<String> charIndexer) {
		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();
		return (markovVerticalOffset ? 
			new CachingEmissionModelExplicitOffsetFactory(charIndexer, paddingMinWidth, paddingMaxWidth, emissionInnerLoop) : 
			new CachingEmissionModelFactory(charIndexer, paddingMinWidth, paddingMaxWidth, emissionInnerLoop));
	}

	private EmissionCacheInnerLoop getEmissionInnerLoop() {
		switch (emissionEngine) {
			case DEFAULT: return new DefaultInnerLoop(numEmissionCacheThreads);
			case OPENCL: return new OpenCLInnerLoop(numEmissionCacheThreads);
			case CUDA: return new CUDAInnerLoop(numEmissionCacheThreads, cudaDeviceID);
		}
		throw new RuntimeException("emissionEngine=" + emissionEngine + " not supported");
	}

	private MultiDocumentEvaluator makeEvalSetEvaluator(Indexer<String> charIndexer, DecoderEM decoderEM, SingleDocumentEvaluator documentEvaluator) {
		MultiDocumentEvaluator evalSetEvaluator;
		if (evalInputDocPath != null) {
			List<Document> evalDocuments = LazyRawImageLoader.loadDocuments(evalInputDocPath, evalExtractedLinesPath, evalNumDocs, 0, true, uniformLineHeight, binarizeThreshold, crop);
			evalSetEvaluator = new BasicMultiDocumentEvaluator(evalDocuments, evalInputDocPath, outputPath, decoderEM, documentEvaluator, charIndexer);
		}
		else {
			evalSetEvaluator = new MultiDocumentEvaluator.NoOpMultiDocumentEvaluator();
		}
		return evalSetEvaluator;
	}

}
