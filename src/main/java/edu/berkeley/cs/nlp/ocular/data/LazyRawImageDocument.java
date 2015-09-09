package edu.berkeley.cs.nlp.ocular.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.berkeley.cs.nlp.ocular.image.ImageUtils;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.image.Visualizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Binarizer;
import edu.berkeley.cs.nlp.ocular.preprocessing.Cropper;
import edu.berkeley.cs.nlp.ocular.preprocessing.LineExtractor;
import edu.berkeley.cs.nlp.ocular.preprocessing.Straightener;
import fileio.f;

/**
 * A document that reads a file only as it is needed (and then stores
 * the contents in memory for later use).
 * 
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public abstract class LazyRawImageDocument implements ImageLoader.Document {
	private final String inputPath;
	private final int lineHeight;
	private final double binarizeThreshold;
	private final boolean crop;

	private PixelType[][][] observations = null;

	private String preextractedLinesPath = null;

	public LazyRawImageDocument(String inputPath, int lineHeight, double binarizeThreshold, boolean crop, String preextractedLinesPath) {
		this.inputPath = inputPath;
		this.lineHeight = lineHeight;
		this.binarizeThreshold = binarizeThreshold;
		this.crop = crop;
		this.preextractedLinesPath = preextractedLinesPath;
	}

	final public PixelType[][][] loadLineImages() {
	  if (observations == null) { // file has been loaded in this Ocular run
		    if (preextractedLinesPath == null) {
		    	doLoadObservationsFromFile(); // load data from original file
		    }
		    else {
		      if (extractionFilesPresent()) { // file has been loaded in a previous Ocular run
		      	doLoadObservationsFromExtractionFiles(); // load data from original file
		      }
		      else {
		      	doLoadObservationsFromFile(); // load data from original file
        		doWriteExtractedLines(); // write extracted lines to files so they don't have to be re-extracted next time
		      }
	      }
    }
	  return observations;
	}

	private void doLoadObservationsFromFile() {
		BufferedImage bi = doLoadBufferedImage();
		double[][] levels = ImageUtils.getLevels(bi);
		double[][] rotLevels = Straightener.straighten(levels);
		double[][] cropLevels = crop ? Cropper.crop(rotLevels, binarizeThreshold) : rotLevels;
		Binarizer.binarizeGlobal(binarizeThreshold, cropLevels);
		List<double[][]> lines = LineExtractor.extractLines(cropLevels);
		observations = new PixelType[lines.size()][][];
		for (int i = 0; i < lines.size(); ++i) {
			observations[i] = imageToObservation(ImageUtils.makeImage(lines.get(i)), lineHeight);
		}
	}
	
	private void doLoadObservationsFromExtractionFiles() {
		System.out.println("Loading pre-extracted line images from " + leLineDir());
		final Pattern pattern = Pattern.compile("line(\\d+)\\." + ext());
		File[] lineImageFiles = new File(leLineDir()).listFiles(new FilenameFilter() {
			public boolean accept(File dir, String name) {
				return pattern.matcher(name).matches();
			}
		});
		if (lineImageFiles == null) throw new RuntimeException("lineImageFiles is null");
		if (lineImageFiles.length == 0) throw new RuntimeException("lineImageFiles.length == 0");
		Arrays.sort(lineImageFiles);

		observations = new PixelType[lineImageFiles.length][][];
		for (int i = 0; i < lineImageFiles.length; ++i) {
			Matcher m = pattern.matcher(lineImageFiles[i].getName());
			if (m.find() && Integer.valueOf(m.group(1)) != i) throw new RuntimeException("Trying to load lines from "+leLineDir()+" but the file for line "+i+" is missing (first file seems to be "+Integer.valueOf(m.group(1))+").");
			String lineImageFile = fullLeLinePath(i);
			System.out.println("    Loading pre-extracted line from " + lineImageFile);
			try {
				observations[i] = imageToObservation(f.readImage(lineImageFile), i);
			}
			catch (Exception e) {
				throw new RuntimeException("Couldn't read line image from: " + lineImageFile, e);
			}
		}
	}
	
	private PixelType[][] imageToObservation(BufferedImage image, int lineNum) {
		if (lineHeight >= 0) {
			return ImageUtils.getPixelTypes(ImageUtils.resampleImage(image, lineHeight));
		}
		else {
			return ImageUtils.getPixelTypes(image);
		}
	}

	private void doWriteExtractedLines() {
		String multilineExtractionImagePath = multilineExtractionImagePath();
		System.out.println("Writing file line-extraction image to: " + multilineExtractionImagePath);
		new File(multilineExtractionImagePath).getParentFile().mkdirs();
		f.writeImage(multilineExtractionImagePath, Visualizer.renderLineExtraction(observations));
		
		// Write individual line files
		new File(leLineDir()).mkdirs();
		for (int l = 0; l < observations.length; ++l) {
			PixelType[][] observationLine = observations[l];
			String linePath = fullLeLinePath(l);
			System.out.println("  Writing individual line-extraction image to: " + linePath);
			f.writeImage(linePath, Visualizer.renderLineExtraction(observationLine));
		}
	}
	
	private boolean extractionFilesPresent() {
		return new File(fullLeLinePath(0)).exists();
	}
	
	private String multilineExtractionImagePath() { return fullLePreExt() + "." + ext(); }
	private String leLineDir() { return fullLePreExt() + "_" + ext(); }
	private String fileParent() { return FileUtil.removeCommonPathPrefixOfParents(new File(inputPath), file())._2; }
	private String fullLePreExt() { return preextractedLinesPath + "/" + fileParent() + "/" + preext() + "-line_extract"; }
	private String fullLeLinePath(int lineNum) { return String.format(leLineDir() + "/line%02d." + ext(), lineNum); }
	
	abstract protected File file();
	abstract protected BufferedImage doLoadBufferedImage();
	abstract protected String preext();
	abstract protected String ext();
	
}
