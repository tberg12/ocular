package edu.berkeley.cs.nlp.ocular.data;

import java.awt.image.BufferedImage;
import java.io.File;

import edu.berkeley.cs.nlp.ocular.util.FileUtil;

/**
 * A document that reads a page from a pdf file only as it is needed
 * (and then stores the contents in memory for later use).
 * 
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class LazyRawPdfImageDocument extends LazyRawImageDocument {
	private final File pdfFile;
	private final int pageNumber; // starts at one!

	public LazyRawPdfImageDocument(File pdfFile, int pageNumber, String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String extractedLinesPath) {
		super(inputPath, lineHeight, binarizeThreshold, crop, extractedLinesPath);
		this.pdfFile = pdfFile;
		this.pageNumber = pageNumber;
	}

	protected BufferedImage doLoadBufferedImage() {
		System.out.println("Extracting text line images from " + pdfFile + ", page " + pageNumber);
		return PdfImageReader.readPdfPageAsImage(pdfFile, pageNumber);
	}
	
	protected File file() { return pdfFile; }
	protected String preext() { return new File(baseName()).getName(); }
	protected String ext() { return "png"; }
	
	public String baseName() {
		return FileUtil.withoutExtension(pdfFile.getPath()) + "_pdf_page" + String.format("%05d", pageNumber);
	}

}
