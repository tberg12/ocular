package edu.berkeley.cs.nlp.ocular.data;

import java.io.File;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.image.Visualizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Binarizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Cropper;
import edu.berkeley.cs.nlp.ocular.preprocessing.LineExtractor;
import edu.berkeley.cs.nlp.ocular.preprocessing.Straightener;
import fileio.f;

/**
 * A document that reads a page from a pdf file only as it is needed 
 * (and then stores the contents in memory for later use).
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class LazyRawPdfImageDocument implements ImageLoader.Document {
	private final File pdfFile;
	private final int pageNumber; // starts at one!
	private final String inputPath;
	private final int lineHeight;
	private final double binarizeThreshold;
	private final boolean crop;

	private PixelType[][][] observations = null;

	private String lineExtractionImageOutputPath = null;

	public LazyRawPdfImageDocument(File pdfFile, int pageNumber, String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String lineExtractionImageOutputPath) {
		this.pdfFile = pdfFile;
		this.pageNumber = pageNumber;
		this.inputPath = inputPath;
		this.lineHeight = lineHeight;
		this.binarizeThreshold = binarizeThreshold;
		this.crop = crop;
		this.lineExtractionImageOutputPath = lineExtractionImageOutputPath;
	}

	public PixelType[][][] loadLineImages() {
		if (observations == null) {
			System.out.println("Extracting text line images from " + pdfFile + ", page " + pageNumber);
			double[][] levels = ImageUtils.getLevels(PdfImageReader.readPdfPageAsImage(pdfFile, pageNumber));
			double[][] rotLevels = Straightener.straighten(levels);
			double[][] cropLevels = crop ? Cropper.crop(rotLevels, binarizeThreshold) : rotLevels;
			Binarizer.binarizeGlobal(binarizeThreshold, cropLevels);
			List<double[][]> lines = LineExtractor.extractLines(cropLevels);
			observations = new PixelType[lines.size()][][];
			for (int i = 0; i < lines.size(); ++i) {
				if (lineHeight >= 0) {
					observations[i] = ImageUtils.getPixelTypes(ImageUtils.resampleImage(ImageUtils.makeImage(lines.get(i)), lineHeight));
				}
				else {
					observations[i] = ImageUtils.getPixelTypes(ImageUtils.makeImage(lines.get(i)));
				}
			}

			if (lineExtractionImageOutputPath != null) {
				String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), pdfFile)._2;
				String preext = new File(baseName()).getName();
				String ext = "jpg";
				
				String lineExtractionImagePath = lineExtractionImageOutputPath + "/" + fileParent + "/" + preext + "-line_extract." + ext;
				System.out.println("Writing line-extraction image to: " + lineExtractionImagePath);
				new File(lineExtractionImagePath).getParentFile().mkdirs();
				f.writeImage(lineExtractionImagePath, Visualizer.renderLineExtraction(observations));
			}
		}
		return observations;
	}

	public String[][] loadLineText() {
		return null;
	}

	public String baseName() {
		return FileUtil.withoutExtension(pdfFile.getPath()) + "_pdf_page" + String.format("%05d", pageNumber);
	}

}
