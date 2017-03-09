package edu.berkeley.cs.nlp.ocular.main;

import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.train.FontTrainer;
import edu.berkeley.cs.nlp.ocular.train.TrainingRestarter;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import tberg.murphy.fig.Option;
import tberg.murphy.indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class TrainFont extends FonttrainTranscribeShared {

	@Option(gloss = "Number of iterations of EM to use for font learning.")
	public static int numEMIters = 3;
	
	@Option(gloss = "If true, the font trainer will find the latest completed iteration in the outputPath and load it in order to pick up training from that point.  Convenient if a training run crashes when only partially completed.")
	public static boolean continueFromLastCompleteIteration = false;

	@Option(gloss = "When using -evalInputDocPath, the font trainer will perform an evaluation every `evalFreq` iterations. Default: Evaluate only after all iterations have completed.")
	public static int evalFreq = Integer.MAX_VALUE; 
	

	public static void main(String[] args) {
		System.out.println("TrainFont");
		TrainFont main = new TrainFont();
		main.doMain(main, args);
	}

	protected void validateOptions() {
		super.validateOptions();
		
		if (numEMIters <= 0) new IllegalArgumentException("-numEMIters must be a positive number.");

		if (outputFontPath == null) throw new IllegalArgumentException("-outputFontPath is required for font training.");
	}

	public void run(List<String> commandLineArgs) {
		Set<OutputFormat> outputFormats = parseOutputFormats();
		
		CodeSwitchLanguageModel initialLM = loadInputLM();
		Font initialFont = loadInputFont();
		BasicGlyphSubstitutionModelFactory gsmFactory = makeGsmFactory(initialLM);
		GlyphSubstitutionModel initialGSM = loadInitialGSM(gsmFactory);
		
		Indexer<String> charIndexer = initialLM.getCharacterIndexer();
		Indexer<String> langIndexer = initialLM.getLanguageIndexer();
		
		DecoderEM decoderEM = makeDecoder(charIndexer);

		boolean evalCharIncludesDiacritic = true;
		SingleDocumentEvaluatorAndOutputPrinter documentOutputPrinterAndEvaluator = new BasicSingleDocumentEvaluatorAndOutputPrinter(charIndexer, langIndexer, allowGlyphSubstitution, evalCharIncludesDiacritic, commandLineArgs);
		
		List<String> inputDocPathList = getInputDocPathList();
		List<Document> inputDocuments = LazyRawImageLoader.loadDocuments(inputDocPathList, extractedLinesPath, numDocs, numDocsToSkip, uniformLineHeight, binarizeThreshold, crop);
		if (inputDocuments.isEmpty()) throw new NoDocumentsFoundException();
		if (updateDocBatchSize > 0 && inputDocuments.size() < updateDocBatchSize) throw new RuntimeException("The number of available documents is less than -updateDocBatchSize!");
		
		String newInputDocPath = FileUtil.lowestCommonPath(inputDocPathList);

		MultiDocumentTranscriber evalSetEvaluator = makeEvalSetEvaluator(charIndexer, decoderEM, documentOutputPrinterAndEvaluator);
		new FontTrainer().trainFont(
				inputDocuments,  
				initialFont, initialLM, initialGSM,
				continueFromLastCompleteIteration ? new TrainingRestarter() : null,
				outputFontPath, outputLmPath, outputGsmPath,
				decoderEM,
				gsmFactory, documentOutputPrinterAndEvaluator,
				numEMIters, updateDocBatchSize > 0 ? updateDocBatchSize : inputDocuments.size(), false, true,
				numMstepThreads,
				newInputDocPath, outputPath, outputFormats,
				evalSetEvaluator, evalFreq, evalBatches,
				skipFailedDocs);
	}

}
