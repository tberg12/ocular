package edu.berkeley.cs.nlp.ocular.main;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import tberg.murphy.fig.Option;
import tberg.murphy.fileio.f;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public abstract class LineExtractionOptions extends OcularRunnable {

	// Main Options
	
	@Option(gloss = "Path to the directory that contains the input document images. The entire directory will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Files will be processed in lexicographical order.")
	public static String inputDocPath = null; // Either inputDocPath or inputDocListPath is required.
	
	@Option(gloss = "Path to a file that contains a list of paths to images files that should be used.  The file should contain one path per line. These paths will be searched in order.  Each path may point to either a file or a directory, which will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Paths will be processed in the order given in the file, and each path will be searched in lexicographical order.")
	public static String inputDocListPath = null; // Either inputDocPath or inputDocListPath is required.

	@Option(gloss = "Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents. Default: Use all documents.")
	public static int numDocs = Integer.MAX_VALUE;

	@Option(gloss = "Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.")
	public static int numDocsToSkip = 0;

	@Option(gloss = "Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.")
	public static String extractedLinesPath = null; // Don't read or write line image files.
	
	// Line Extraction Options
	
	@Option(gloss = "Quantile to use for pixel value thresholding. (High values mean more black pixels.)")
	public static double binarizeThreshold = 0.12;

	@Option(gloss = "Crop pages?")
	public static boolean crop = true;

	@Option(gloss = "Scale all lines to have the same height?")
	public static boolean uniformLineHeight = true;
	

	
	protected void validateOptions() {
		if ((inputDocPath == null) == (inputDocListPath == null)) throw new IllegalArgumentException("Either -inputDocPath or -inputDocListPath is required.");
		if (inputDocPath != null)
			for (String path : inputDocPath.split("[\\s,;:]+"))
				if (!new File(path).exists()) throw new IllegalArgumentException("inputDocPath "+path+" does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (inputDocListPath != null && !new File(inputDocListPath).exists()) throw new IllegalArgumentException("-inputDocListPath "+inputDocListPath+" does not exist [looking in "+(new File(".").getAbsolutePath())+"]");
		if (numDocsToSkip < 0) throw new IllegalArgumentException("-numDocsToSkip must be >= 0.  Was "+numDocsToSkip+".");
	}
	
	protected static List<String> getInputDocPathList() {
		return inputDocPath != null ? Arrays.asList(inputDocPath.split("[\\s+,;:]")) : f.readLines(inputDocListPath);
	}


}
