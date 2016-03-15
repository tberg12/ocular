package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter.diplomaticTranscriptionOutputFile;
import static edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter.makeOutputFilenameBase;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicMultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.train.FontTrainer;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import fig.Option;
import fig.OptionsParser;
import indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Transcribe extends FonttrainTranscribeShared implements Runnable {

	// ##### Main Options
	
//	@Option(gloss = "Path to the directory that contains the input document images. The entire directory will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Files will be processed in lexicographical order.")
//	public static String inputDocPath = null; // Either inputDocPath or inputDocListPath is required.
	
//	@Option(gloss = "Path to a file that contains a list of paths to images files that should be used.  The file should contain one path per line. These paths will be searched in order.  Each path may point to either a file or a directory, which will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Paths will be processed in the order given in the file, and each path will be searched in lexicographical order.")
//	public static String inputDocListPath = null; // Either inputDocPath or inputDocListPath is required.

//	@Option(gloss = "Path of the directory that will contain output transcriptions.")
//	public static String outputPath = null; // Required.
//
//	@Option(gloss = "Path to the input language model file.")
//	public static String inputLmPath = null; // Required.
//
//	@Option(gloss = "Path of the input font file.")
//	public static String inputFontPath = null; // Required.

//	@Option(gloss = "Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents. Default: Use all documents.")
//	public static int numDocs = Integer.MAX_VALUE;

//	@Option(gloss = "Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.")
//	public static int numDocsToSkip = 0;

//	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
//	public static String extractedLinesPath = null; // Don't read or write line image files.

	@Option(gloss = "If true, for each doc the outputPath will be checked for an existing transcription and if one is found then the document will be skipped.")
	public static boolean skipAlreadyTranscribedDocs = false;
	
	// ##### Font Learning Options
	
	@Option(gloss = "Update the font during transcription based on the new input documents?")
	public static boolean updateFont = false;
	
	// The following options are only relevant if updateFont is set to "true".

//	@Option(gloss = "Path to write the learned font file to. Required if updateFont is set to true, otherwise ignored.")
//	public static String outputFontPath = null;

//	@Option(gloss = "Number of documents to process for each parameter update.  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with updateFont=true.  Default: Update only after each full pass over the document set.")
//	public static int updateDocBatchSize = -1;

	// ##### Language Model Re-training Options
	
//	@Option(gloss = "Should the language model be updated during font training?")
//	public static boolean updateLM = false;
//	
//	@Option(gloss = "Path to write the retrained language model file to. Required if updateLM is set to true, otherwise ignored.")
//	public static String outputLmPath = null;

	// ##### Glyph Substitution Model Options
	
//	@Option(gloss = "Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions.")
//	public static boolean allowGlyphSubstitution = false;
	
	// The following options are only relevant if allowGlyphSubstitution is set to "true".
	
//	@Option(gloss = "Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.) Default: Don't use a pre-initialized GSM. (Learn one from scratch).")
//	public static String inputGsmPath = null;
//
//	@Option(gloss = "Exponent on GSM scores.")
//	public static double gsmPower = 4.0;
//
//	@Option(gloss = "The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions.")
//	public static double gsmNoCharSubPrior = 0.9;
//
//	@Option(gloss = "Should the GSM be allowed to elide letters even without the presence of an elision-marking tilde?")
//	public static boolean gsmElideAnything = false;
//	
//	@Option(gloss = "Should the glyph substitution model be trained (or updated) along with the font? (Only relevant if allowGlyphSubstitution is set to true.)")
//	public static boolean updateGsm = false;
//	
//	@Option(gloss = "Path to write the retrained glyph substitution model file to. Required if updateGsm is set to true, otherwise ignored.")
//	public static String outputGsmPath = null;
//	
//	@Option(gloss = "The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.")
//	public static double gsmSmoothingCount = 1.0;
//	
//	@Option(gloss = "gsmElisionSmoothingCountMultiplier.")
//	public static double gsmElisionSmoothingCountMultiplier = 100.0;
	
	// ##### Line Extraction Options
	
//	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
//	public static double binarizeThreshold = 0.12;

//	@Option(gloss = "Crop pages?")
//	public static boolean crop = true;

//	@Option(gloss = "Scale all lines to have the same height?")
//	public static boolean uniformLineHeight = true;

	// ##### Miscellaneous Options
	
//	@Option(gloss = "Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.")
//	public static EmissionCacheInnerLoopType emissionEngine = EmissionCacheInnerLoopType.DEFAULT; // Default: DEFAULT
//
//	@Option(gloss = "Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)")
//	public static int beamSize = 10;
//
//	@Option(gloss = "GPU ID when using CUDA emission engine.")
//	public static int cudaDeviceID = 0;
//
//	@Option(gloss = "Number of threads to use for LFBGS during m-step.")
//	public static int numMstepThreads = 8;
//
//	@Option(gloss = "Number of threads to use during emission cache compuation. (Only has effect when emissionEngine is set to DEFAULT.)")
//	public static int numEmissionCacheThreads = 8;
//
//	@Option(gloss = "Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)")
//	public static int numDecodeThreads = 8;
//
//	@Option(gloss = "Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)")
//	public static int decodeBatchSize = 32;
//
//	@Option(gloss = "Min horizontal padding between characters in pixels. (Best left at default value.)")
//	public static int paddingMinWidth = 1;
//
//	@Option(gloss = "Max horizontal padding between characters in pixels (Best left at default value.)")
//	public static int paddingMaxWidth = 5;
//
//	@Option(gloss = "Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)")
//	public static boolean markovVerticalOffset = false;
//
//	@Option(gloss = "A language model to be used to assign diacritics to the transcription output.")
//	public static boolean allowLanguageSwitchOnPunct = true;
	
	// ##### Options used if evaluation should be performed during training
	
//	@Option(gloss = "When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
//	public static String evalInputDocPath = null; // Do not evaluate during font training.

	// The following options are only relevant if a value was given to -evalInputDocPath.
	
//	@Option(gloss = "When using -evalInputDocPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
//	public static String evalExtractedLinesPath = null; // Don't read or write line image files. 
//
//	@Option(gloss = "When using -evalInputDocPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents. Default: Use all documents in the specified path.")
//	public static int evalNumDocs = Integer.MAX_VALUE;
//
//	@Option(gloss = "When using -evalInputDocPath, the font trainer will perform an evaluation every `evalFreq` iterations. Default: Evaluate only after all iterations have completed.")
//	public static int evalFreq = Integer.MAX_VALUE; 
//	
//	@Option(gloss = "When using -evalInputDocPath, on iterations in which we run the evaluation, should the evaluation be run after each batch, as determined by -updateDocBatchSize (in addition to after each iteration)?")
//	public static boolean evalBatches = false;


	public static void main(String[] args) {
		Transcribe main = new Transcribe();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		validateOptions();
		main.run();
	}

	protected static void validateOptions() {
		FonttrainTranscribeShared.validateOptions();
		
		if (updateFont && outputFontPath == null) throw new IllegalArgumentException("-outputFontPath required when -updateFont is true.");
		if (!updateFont && outputFontPath != null) throw new IllegalArgumentException("-outputFontPath not permitted when -updateFont is false.");

		if (!(updateFont == (outputFontPath != null))) throw new IllegalArgumentException("-updateFont is not as expected");
	}

	public void run() {
		CodeSwitchLanguageModel initialLM = loadInputLM();
		Font initialFont = loadInputFont();
		BasicGlyphSubstitutionModelFactory gsmFactory = makeGsmFactory(initialLM);
		GlyphSubstitutionModel initialGSM = loadInitialGSM(gsmFactory);
		
		Indexer<String> charIndexer = initialLM.getCharacterIndexer();
		Indexer<String> langIndexer = initialLM.getLanguageIndexer();
		
		DecoderEM decoderEM = makeDecoder(charIndexer);

		boolean evalCharIncludesDiacritic = true;
		SingleDocumentEvaluatorAndOutputPrinter documentOutputPrinterAndEvaluator = new BasicSingleDocumentEvaluatorAndOutputPrinter(charIndexer, langIndexer, allowGlyphSubstitution, evalCharIncludesDiacritic);
		
		List<String> inputDocPathList = getInputDocPathList();
		List<Document> inputDocuments = LazyRawImageLoader.loadDocuments(inputDocPathList, extractedLinesPath, numDocs, numDocsToSkip, uniformLineHeight, binarizeThreshold, crop);
		if (inputDocuments.isEmpty()) throw new RuntimeException("No documents given!");
		String newInputDocPath = FileUtil.lowestCommonPath(inputDocPathList);

		if (skipAlreadyTranscribedDocs) {
			for (Iterator<Document> itr = inputDocuments.iterator(); itr.hasNext(); ) {
				Document doc = itr.next();
				String docTranscriptionPath = diplomaticTranscriptionOutputFile(makeOutputFilenameBase(doc, newInputDocPath, outputPath));
				if (new File(docTranscriptionPath).exists()) {
					System.out.println("  Skipping " + doc.baseName() + " since it was already transcribed: ["+docTranscriptionPath+"]");
					itr.remove();
				}
			}
		}

		if (outputFontPath != null) {
			//
			// Update the font as we transcribe
			//
			MultiDocumentTranscriber evalSetEvaluator = makeEvalSetEvaluator(charIndexer, decoderEM, documentOutputPrinterAndEvaluator);
			new FontTrainer().doFontTrainPass(0,
					inputDocuments,  
					initialFont, initialLM, initialGSM,
					outputFontPath, outputLmPath, outputGsmPath,
					decoderEM,
					gsmFactory, documentOutputPrinterAndEvaluator,
					0, updateDocBatchSize > 0 ? updateDocBatchSize : inputDocuments.size(), true, false,
					numMstepThreads,
					newInputDocPath, outputPath,
					evalSetEvaluator, evalFreq, evalBatches);
		}
		else {
			//
			// Transcribe with fixed parameters
			//
			System.out.println("Transcribing input data      " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			MultiDocumentTranscriber transcriber = new BasicMultiDocumentTranscriber(inputDocuments, newInputDocPath, outputPath, decoderEM, documentOutputPrinterAndEvaluator, charIndexer);
			transcriber.transcribe(initialFont, initialLM, initialGSM);
		}
		
		System.out.println("Completed.");
	}

}
