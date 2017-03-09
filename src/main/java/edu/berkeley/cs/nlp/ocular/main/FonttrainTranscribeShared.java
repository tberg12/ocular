package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.*;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.makeList;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicMultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.gsm.NoSubGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.em.CUDAInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.em.DefaultInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.em.EmissionCacheInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.em.JOCLInnerLoop;
import edu.berkeley.cs.nlp.ocular.model.emission.CachingEmissionModel.CachingEmissionModelFactory;
import edu.berkeley.cs.nlp.ocular.model.emission.CachingEmissionModelExplicitOffset.CachingEmissionModelExplicitOffsetFactory;
import edu.berkeley.cs.nlp.ocular.model.emission.EmissionModel.EmissionModelFactory;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import tberg.murphy.fig.Option;
import tberg.murphy.indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public abstract class FonttrainTranscribeShared extends LineExtractionOptions {

	@Option(gloss = "Path of the directory that will contain output transcriptions.")
	public static String outputPath = null; // Required.

	public static enum OutputFormat { DIPL, NORM, NORMLINES, COMP, HTML, ALTO, WHITESPACE };
	@Option(gloss = "Output formats to be generated. Choose from one or multiple of {dipl,norm,normlines,comp,html,alto}, comma-separated.  dipl = diplomatic, norm = normalized (lines joined), normlines = normalized (separate lines), comp = comparisons.  Default: dipl,norm if -allowGlyphSubstitution=true; dipl otherwise.")
	public static String outputFormats = "";

	@Option(gloss = "Path to the input language model file.")
	public static String inputLmPath = null; // Required.

	@Option(gloss = "Path of the input font file.")
	public static String inputFontPath = null; // Required.
	
	// ##### Font Learning Options

	@Option(gloss = "Path to write the learned font file to. Required if updateFont is set to true, otherwise ignored.")
	public static String outputFontPath = null;

	@Option(gloss = "Number of documents to process for each parameter update.  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with updateFont=true.  Default: Update only after each full pass over the document set.")
	public static int updateDocBatchSize = -1;

	// ##### Language Model Re-training Options
	
	@Option(gloss = "Should the language model be updated along with the font?")
	public static boolean updateLM = false;
	
	@Option(gloss = "Path to write the retrained language model file to. Required if updateLM is set to true, otherwise ignored.")
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
	
	@Option(gloss = "Should the glyph substitution model be trained (or updated) along with the font? (Only relevant if allowGlyphSubstitution is set to true.)")
	public static boolean updateGsm = false;
	
	@Option(gloss = "Path to write the retrained glyph substitution model file to. Required if updateGsm is set to true, otherwise ignored.")
	public static String outputGsmPath = null;
	
	@Option(gloss = "The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.")
	public static double gsmSmoothingCount = 1.0;
	
	@Option(gloss = "gsmElisionSmoothingCountMultiplier.")
	public static double gsmElisionSmoothingCountMultiplier = 100.0;

	// ##### Miscellaneous Options

	@Option(gloss = "Should documents that cause errors be skipped instead of stopping the whole program?")
	public static boolean skipFailedDocs = false;
	
	public static enum EmissionCacheInnerLoopType { DEFAULT, OPENCL, CUDA };
	@Option(gloss = "Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT; // Default: DEFAULT

	@Option(gloss = "Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)")
	public static int beamSize = 10;

	@Option(gloss = "GPU ID when using CUDA emission engine.")
	public static int cudaDeviceID = 0;

	@Option(gloss = "Number of threads to use for LFBGS during m-step.")
	public static int numMstepThreads = 8;

	@Option(gloss = "Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.)")
	public static int numEmissionCacheThreads = 8;

	@Option(gloss = "Number of threads to use for decoding. (More thread may increase speed, but may cause a loss of continuity across lines.)")
	public static int numDecodeThreads = 1;

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
	
	@Option(gloss = "When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`). (Only relevant if updateFont is set to true.)")
	public static String evalInputDocPath = null; // Do not evaluate during font training.

	// The following options are only relevant if a value was given to -evalInputDocPath.
	
	@Option(gloss = "When using -evalInputDocPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String evalExtractedLinesPath = null; // Don't read or write line image files. 

	@Option(gloss = "When using -evalInputDocPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents. Default: Use all documents in the specified path.")
	public static int evalNumDocs = Integer.MAX_VALUE;

	@Option(gloss = "When using -evalInputDocPath, on iterations in which we run the evaluation, should the evaluation be run after each batch, as determined by -updateDocBatchSize (in addition to after each iteration)?")
	public static boolean evalBatches = false;
	
	//
	
	protected static Set<OutputFormat> parseOutputFormats() {
		Set<OutputFormat> formats = new HashSet<OutputFormat>();
		List<String> invalidFormats = new ArrayList<String>();
		for (String fs: outputFormats.replaceAll("\\s+", "").split(",")) {
			if (!fs.isEmpty()) {
				String fsu = fs.toUpperCase();
				OutputFormat of = null;
				try {
					of = OutputFormat.valueOf(fsu);
				}
				catch (IllegalArgumentException e) {
					invalidFormats.add(fs);
				}
				if (of != null) {
					if ((of == NORM || of == NORMLINES) && !allowGlyphSubstitution)
						throw new IllegalArgumentException("-outputFormats 'norm' and 'normlines' are not valid if -allowGlyphSubstitution is false");
					formats.add(of);
				}
			}
		}
		if (!invalidFormats.isEmpty()) {
			throw new IllegalArgumentException("Invalid output formats: {"+StringHelper.join(invalidFormats, ", ")+"}");
		}
		if (formats.isEmpty()) {
			formats.add(DIPL);
			if (allowGlyphSubstitution)
				formats.add(NORM);
		}
		return formats;
	}
	
	protected void validateOptions() {
		super.validateOptions();

		if (outputPath == null) throw new IllegalArgumentException("-outputPath not set");
		parseOutputFormats();
		
		if (inputFontPath == null) throw new IllegalArgumentException("-inputFontPath is required");
		if (!new File(inputFontPath).exists()) throw new RuntimeException("inputFontPath " + inputFontPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");

		if (inputLmPath == null) throw new IllegalArgumentException("-inputLmPath is required");
		if (inputLmPath != null && !new File(inputLmPath).exists()) throw new RuntimeException("inputLmPath " + inputLmPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (updateLM && outputLmPath == null) throw new IllegalArgumentException("-outputLmPath required when -updateLM is true.");
		if (!updateLM && outputLmPath != null) throw new IllegalArgumentException("-outputLmPath not permitted when -updateLM is false.");
		if (outputLmPath != null && outputFontPath == null) throw new IllegalArgumentException("It is not possible to retrain the LM (-updateLM=true) when not retraining the font (-updateFont=false).");

		if (updateGsm && !allowGlyphSubstitution) throw new IllegalArgumentException("-updateGsm not permitted if -allowGlyphSubstitution is false.");
		if (inputGsmPath != null && !new File(inputGsmPath).exists()) throw new RuntimeException("inputGsmPath " + inputGsmPath + " does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (inputGsmPath != null && !allowGlyphSubstitution) throw new IllegalArgumentException("-inputGsmPath not permitted if -allowGlyphSubstitution is false.");
		if (outputGsmPath != null && !allowGlyphSubstitution) throw new IllegalArgumentException("-outputGsmPath not permitted if -allowGlyphSubstitution is false.");
		if (updateGsm && outputGsmPath == null) throw new IllegalArgumentException("-outputGsmPath required when -updateGsm is true.");
		if (!updateGsm && outputGsmPath != null) throw new IllegalArgumentException("-outputGsmPath not permitted when -updateGsm is false.");
		if (allowGlyphSubstitution && inputGsmPath == null && outputGsmPath == null) throw new IllegalArgumentException("If -allowGlyphSubstitution=true, either an -inputGsmPath must be given, or a GSM must be trained by giving an -outputGsmPath.");
		if (outputGsmPath != null && outputFontPath == null) throw new IllegalArgumentException("It is not possible to retrain the GSM (-updateGsm=true) when not retraining the font (-updateFont=false).");

		if (evalExtractedLinesPath != null && evalInputDocPath == null) throw new IllegalArgumentException("-evalExtractedLinesPath not permitted without -evalInputDocPath.");

		// Make the output directory if it doesn't exist yet
		new File(outputPath).mkdirs();
		
		//
		
		if (!(updateLM == (outputLmPath != null))) throw new IllegalArgumentException("-updateLM is not as expected");
		if (!(updateGsm == (outputGsmPath != null))) throw new IllegalArgumentException("-updateGsm is not as expected");
		if (!(allowGlyphSubstitution == (inputGsmPath != null || outputGsmPath != null))) throw new IllegalArgumentException("-allowGlyphSubstitution is not as expected");
	}
	

	protected static CodeSwitchLanguageModel loadInputLM() {
		System.out.println("Loading initial LM from " + inputLmPath);
		CodeSwitchLanguageModel codeSwitchLM = InitializeLanguageModel.readCodeSwitchLM(inputLmPath);

		//print some useful info
		{
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
		}

		return codeSwitchLM;
	}

	protected static Font loadInputFont() {
		System.out.println("Loading font from " + inputFontPath);
		return InitializeFont.readFont(inputFontPath);
	}
	
	protected static BasicGlyphSubstitutionModelFactory makeGsmFactory(CodeSwitchLanguageModel lm) {
		Indexer<String> charIndexer = lm.getCharacterIndexer();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		Set<Integer>[] activeCharacterSets = makeActiveCharacterSets(lm);
		return new BasicGlyphSubstitutionModelFactory(gsmSmoothingCount, gsmElisionSmoothingCountMultiplier, langIndexer, charIndexer, activeCharacterSets, gsmPower, 0, outputPath);
	}

	public static Set<Integer>[] makeActiveCharacterSets(CodeSwitchLanguageModel lm) {
		int numLanguages = lm.getLanguageIndexer().size();
		@SuppressWarnings("unchecked")
		Set<Integer>[] activeCharacterSets = new Set[numLanguages];
		for (int l = 0; l < numLanguages; ++l) activeCharacterSets[l] = lm.get(l).getActiveCharacters();
		return activeCharacterSets;
	}
	
	protected static GlyphSubstitutionModel loadInitialGSM(BasicGlyphSubstitutionModelFactory gsmFactory) {
		if (!allowGlyphSubstitution) {
			System.out.println("Glyph substitution not allowed; constructing no-sub GSM.");
			return new NoSubGlyphSubstitutionModel();
		}
		else if (inputGsmPath != null) { // file path given
			System.out.println("Loading initial GSM from " + inputGsmPath);
			return InitializeGlyphSubstitutionModel.readGSM(inputGsmPath);
		}
		else {
			System.out.println("No initial GSM provided; initializing to uniform model.");
			return gsmFactory.uniform();
		}
	}
	
	protected static DecoderEM makeDecoder(Indexer<String> charIndexer) {
		EmissionModelFactory emissionModelFactory = makeEmissionModelFactory(charIndexer);
		return new DecoderEM(emissionModelFactory, allowGlyphSubstitution, gsmNoCharSubPrior, gsmElideAnything, allowLanguageSwitchOnPunct, markovVerticalOffset, beamSize, numDecodeThreads, numMstepThreads, decodeBatchSize);
	}

	protected static EmissionModelFactory makeEmissionModelFactory(Indexer<String> charIndexer) {
		EmissionCacheInnerLoop emissionInnerLoop = getEmissionInnerLoop();
		return (markovVerticalOffset ? 
			new CachingEmissionModelExplicitOffsetFactory(charIndexer, paddingMinWidth, paddingMaxWidth, emissionInnerLoop) : 
			new CachingEmissionModelFactory(charIndexer, paddingMinWidth, paddingMaxWidth, emissionInnerLoop));
	}

	protected static EmissionCacheInnerLoop getEmissionInnerLoop() {
		switch (emissionEngine) {
			case DEFAULT: return new DefaultInnerLoop(numEmissionCacheThreads);
			case OPENCL: return new JOCLInnerLoop(numEmissionCacheThreads);
			case CUDA: return new CUDAInnerLoop(numEmissionCacheThreads, cudaDeviceID);
		}
		throw new RuntimeException("emissionEngine=" + emissionEngine + " not supported");
	}

	protected static MultiDocumentTranscriber makeEvalSetEvaluator(Indexer<String> charIndexer, DecoderEM decoderEM, SingleDocumentEvaluatorAndOutputPrinter documentOutputPrinterAndEvaluator) {
		if (evalInputDocPath != null) {
			List<Document> evalDocuments = LazyRawImageLoader.loadDocuments(evalInputDocPath, evalExtractedLinesPath, evalNumDocs, 0, uniformLineHeight, binarizeThreshold, crop);
			if (evalDocuments.isEmpty()) throw new NoDocumentsFoundException("No evaluation documents found! Checked -evalInputDocPath = "+evalInputDocPath);
			for (Document doc : evalDocuments) {
				if (doc.loadDiplomaticTextLines() == null & doc.loadNormalizedText() == null) 
					throw new RuntimeException("Evaluation document "+doc.baseName()+" has no gold transcriptions.");
			}
			return new BasicMultiDocumentTranscriber(evalDocuments, evalInputDocPath, outputPath, parseOutputFormats(), decoderEM, documentOutputPrinterAndEvaluator, charIndexer, skipFailedDocs);
		}
		else {
			return new MultiDocumentTranscriber.NoOpMultiDocumentTranscriber();
		}
	}
	
}
