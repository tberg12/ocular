package edu.berkeley.cs.nlp.ocular.main;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import fig.OptionsParser;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ExtractLinesOnly extends FonttrainTranscribeShared implements Runnable {

	public static void main(String[] args) {
		ExtractLinesOnly main = new ExtractLinesOnly();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		validateOptions();
		main.run();
	}

	public void run() {
		List<String> inputDocPathList = getInputDocPathList();
		List<Document> inputDocuments = LazyRawImageLoader.loadDocuments(inputDocPathList, extractedLinesPath, numDocs, numDocsToSkip, uniformLineHeight, binarizeThreshold, crop);
		if (inputDocuments.isEmpty()) throw new NoDocumentsFoundException();
		for (Document doc : inputDocuments) {
			doc.loadLineImages();
		}
	}
	
	protected static void validateOptions() {
		FonttrainTranscribeShared.validateOptions();
		
		if (extractedLinesPath == null) throw new IllegalArgumentException("-extractedLinesPath is required.");
	}

}
