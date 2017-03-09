package edu.berkeley.cs.nlp.ocular.main;

import static edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter.diplomaticTranscriptionOutputFile;
import static edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter.makeOutputFilenameBase;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import edu.berkeley.cs.nlp.ocular.eval.BasicMultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.BasicSingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.train.FontTrainer;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import tberg.murphy.fig.Option;
import tberg.murphy.indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Transcribe extends FonttrainTranscribeShared {

	@Option(gloss = "If true, for each doc the outputPath will be checked for an existing transcription and if one is found then the document will be skipped.")
	public static boolean skipAlreadyTranscribedDocs = false;
	
	@Option(gloss = "If true, an exception will be thrown if all of the input documents have already been transcribed (and thus the job has nothing to do).  Ignored unless -skipAlreadyTranscribedDocs=true.")
	public static boolean failIfAllDocsAlreadyTranscribed = false;
	
	@Option(gloss = "Update the font during transcription based on the new input documents?")
	public static boolean updateFont = false;
	

	public static void main(String[] args) {
		System.out.println("Transcribe");
		Transcribe main = new Transcribe();
		main.doMain(main, args);
	}

	protected void validateOptions() {
		super.validateOptions();
		
		if (updateFont && outputFontPath == null) throw new IllegalArgumentException("-outputFontPath is required when -updateFont is true.");
		if (!updateFont && outputFontPath != null) throw new IllegalArgumentException("-outputFontPath not permitted when -updateFont is false.");
		
		if (evalBatches && !updateFont) throw new IllegalArgumentException("-evalBatches doesn't make sense when -updateFont is false.");

		if (!(updateFont == (outputFontPath != null))) throw new IllegalArgumentException("-updateFont is not as expected");
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

		String newInputDocPath = FileUtil.lowestCommonPath(inputDocPathList);
		if (skipAlreadyTranscribedDocs) {
			int numInputDocsBeforeSkipping = inputDocuments.size();
			for (Iterator<Document> itr = inputDocuments.iterator(); itr.hasNext(); ) {
				Document doc = itr.next();
				String docTranscriptionPath = diplomaticTranscriptionOutputFile(makeOutputFilenameBase(doc, newInputDocPath, outputPath));
				if (new File(docTranscriptionPath).exists()) {
					System.out.println("  Skipping " + doc.baseName() + " since it was already transcribed: ["+docTranscriptionPath+"]");
					itr.remove();
				}
			}
			if (inputDocuments.isEmpty()) {
				String msg = "The input path contains "+numInputDocsBeforeSkipping+" documents, but all have already been transcribed, so there is nothing remaining for this job to do.  (This is due to setting -skipAlreadyTranscribedDocs=true.)";
				if (failIfAllDocsAlreadyTranscribed)
					throw new NoDocumentsToProcessException(msg);
				else
					System.out.println("WARNING: "+msg);
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
					newInputDocPath, outputPath, outputFormats,
					evalSetEvaluator, Integer.MAX_VALUE, evalBatches,
					skipFailedDocs);
		}
		else {
			//
			// Transcribe with fixed parameters
			//
			System.out.println("Transcribing input data      " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			MultiDocumentTranscriber transcriber = new BasicMultiDocumentTranscriber(inputDocuments, newInputDocPath, outputPath, outputFormats, decoderEM, documentOutputPrinterAndEvaluator, charIndexer, skipFailedDocs);
			transcriber.transcribe(initialFont, initialLM, initialGSM);
		}
	}

}
