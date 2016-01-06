package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;

/**
 * A dataset loader that reads files only as they are needed (and then stores
 * the contents in memory for later use).
 * 
 * @author Dan Garrette (dhgarrette@gmail.com)
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

	public static List<Document> loadDocuments(String inputPath, String extractedLinesPath, int numDocs, int numDocsToSkip) { return loadDocuments(inputPath, extractedLinesPath, numDocs, numDocsToSkip, true, 0.12, false); }
	public static List<Document> loadDocuments(String inputPath, String extractedLinesPath, int numDocs, int numDocsToSkip, boolean uniformLineHeight, double binarizeThreshold, boolean crop) {
		int lineHeight = uniformLineHeight ? CharacterTemplate.LINE_HEIGHT : -1;
		LazyRawImageLoader loader = new LazyRawImageLoader(inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath);
		List<Document> documents = new ArrayList<Document>();

		List<Document> lazyDocs = loader.readDataset();
		Collections.sort(lazyDocs, new Comparator<Document>() {
			public int compare(Document o1, Document o2) {
				return o1.baseName().compareTo(o2.baseName());
			}
		});
		
		int actualNumDocsToSkip = Math.min(lazyDocs.size(), numDocsToSkip);
		int actualNumDocsToUse = Math.min(lazyDocs.size() - actualNumDocsToSkip, numDocs <= 0 ? Integer.MAX_VALUE : numDocs);
		System.out.println("Using "+actualNumDocsToUse+" documents (skipping "+actualNumDocsToSkip+")");
		for (int docNum = 0; docNum < actualNumDocsToSkip; ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			System.out.println("  Skipping " + lazyDoc.baseName());
		}
		for (int docNum = actualNumDocsToSkip; docNum < actualNumDocsToSkip+actualNumDocsToUse; ++docNum) {
			Document lazyDoc = lazyDocs.get(docNum);
			System.out.println("  Using " + lazyDoc.baseName());
			documents.add(lazyDoc);
		}
		if (actualNumDocsToUse < 1) throw new RuntimeException("No documents given!");
		return documents;
	}
}
