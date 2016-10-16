package edu.berkeley.cs.nlp.ocular.main;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ExtractLinesOnly extends LineExtractionOptions {

	public static void main(String[] args) {
		System.out.println("ExtractLinesOnly");
		ExtractLinesOnly main = new ExtractLinesOnly();
		main.doMain(main, args);
	}
		
	protected void validateOptions() {
		super.validateOptions();
		if (extractedLinesPath == null) throw new IllegalArgumentException("-extractedLinesPath is required.");
	}

	public void run(List<String> commandLineArgs) {
		List<String> inputDocPathList = getInputDocPathList();
		List<Document> inputDocuments = LazyRawImageLoader.loadDocuments(inputDocPathList, extractedLinesPath, numDocs, numDocsToSkip, uniformLineHeight, binarizeThreshold, crop);
		if (inputDocuments.isEmpty()) throw new NoDocumentsFoundException();
		for (Document doc : inputDocuments) {
			doc.loadLineImages();
		}
	}
	
}
