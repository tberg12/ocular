package edu.berkeley.cs.nlp.ocular.main;

import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.LazyRawImageLoader;
import fig.Option;
import fig.OptionsParser;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class ExtractLinesOnly implements Runnable {

	// Main Options
	
	@Option(gloss = "Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).")
	public static String inputPath = null; //"test_img";

	@Option(gloss = "Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents. Default: use all documents")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.  Default: 0")
	public static int numDocsToSkip = 0;

	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String extractedLinesPath = null;
	
	// Line Extraction Options
	
	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = true;

	@Option(gloss = "Scale all lines to have the same height?")
	public static boolean uniformLineHeight = true;
	
	
	public static void main(String[] args) {
		ExtractLinesOnly main = new ExtractLinesOnly();
		OptionsParser parser = new OptionsParser();
		parser.doRegisterAll(new Object[] { main });
		if (!parser.doParse(args)) System.exit(1);
		main.run();
	}

	public void run() {
		if (inputPath == null) throw new IllegalArgumentException("-inputPath not set");
		if (extractedLinesPath == null) throw new IllegalArgumentException("-extractedLinesPath not set");
		List<Document> documents = LazyRawImageLoader.loadDocuments(inputPath, extractedLinesPath, numDocs, numDocsToSkip, false, uniformLineHeight, binarizeThreshold, crop);
		for (Document doc : documents) {
			doc.loadLineImages();
		}
	}

}
