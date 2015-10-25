package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * A dataset loader that reads files only as they are needed (and then stores
 * the contents in memory for later use).
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class LazyRawImageLoader implements ImageLoader {

	private final String inputPath;
	private final int lineHeight;
	private final double binarizeThreshold;
	private final boolean crop;

	private String extractedLinesPath = null;

	public LazyRawImageLoader(String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String extractedLinesPath) {
		this.inputPath = inputPath;
		this.lineHeight = lineHeight;
		this.binarizeThreshold = binarizeThreshold;
		this.crop = crop;
		this.extractedLinesPath = extractedLinesPath;
	}

	public List<Document> readDataset() {
		File dir = new File(inputPath);
		System.out.println("Reading data from [" + dir + "], which " + (dir.exists() ? "exists" : "does not exist"));
		List<File> dirList = FileUtil.recursiveFiles(dir);

		List<Document> docs = new ArrayList<Document>();
		for (File f : dirList) {
			if (f.getName().endsWith(".txt"))
				continue;
			if (f.getName().endsWith(".pdf")) {
				int numPages = PdfImageReader.numPagesInPdf(f);
				for (int pageNumber = 1; pageNumber <= numPages; ++pageNumber) {
					docs.add(new LazyRawPdfImageDocument(f, pageNumber, inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath));
				}
			}
			else {
				docs.add(new LazyRawSingleImageDocument(f, inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath));
			}
		}
		return docs;
	}
}
